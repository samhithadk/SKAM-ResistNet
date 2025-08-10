import os, re, io, time, math, textwrap, warnings, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 👈 headless backend for HF Spaces
import matplotlib.pyplot as plt

import gradio as gr

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
DATA_CANDIDATES = [
    os.getenv("TEM1_DATA_PATH", "tem1_clean.csv"),
    "data/tem1_clean.csv",
    "/data/tem1_clean.csv",
]
UNIPROT_ID = "P62593"  # TEM-1 beta-lactamase
PAFF_BINDER_THRESHOLD = 6.0  # >=6 ~ <=1µM

# -----------------------------
# Small helpers
# -----------------------------
def pAff_to_nM(p):
    # p = -log10(Kd M)  ->  Kd (nM) = 10**(9-p)
    return 10.0 ** (9.0 - float(p))

def fmt_conc(nM):
    if nM < 1e-3:  return f"{nM*1e3:.2f} pM"
    if nM < 1:     return f"{nM:.2f} nM"
    if nM < 1e3:   return f"{nM/1e3:.2f} µM"
    return f"{nM/1e6:.2f} mM"

def conf_label(p):
    if p >= 0.80: return "Likely"
    if p >= 0.60: return "Uncertain"
    return "Unlikely"

def conf_emoji(p):
    if p >= 0.80: return "🟢"
    if p >= 0.60: return "🟡"
    return "🔴"

def _parse_smiles_block(text, limit=100):
    items = [s.strip() for s in re.split(r'[\n,;]+', str(text or "")) if s.strip()]
    return items[:limit]

# -----------------------------
# Load TEM-1 protein and embed
# -----------------------------
print("[boot] Fetching TEM-1 (UniProt %s)" % UNIPROT_ID)
fasta = requests.get(f"https://rest.uniprot.org/uniprotkb/{UNIPROT_ID}.fasta").text
TEM1_SEQ = "".join(line.strip() for line in fasta.splitlines() if not line.startswith(">"))
TEM1_SEQ = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", TEM1_SEQ.upper())
print("[boot] TEM-1 length:", len(TEM1_SEQ))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[boot] Using device:", device)

print("[boot] Loading ESM-2 35M ...")
tok_p = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
mdl_p = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device).eval()

print("[boot] Loading ChemBERTa ...")
tok_l = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
mdl_l = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device).eval()

with torch.inference_mode():
    toks = tok_p(TEM1_SEQ, return_tensors="pt", add_special_tokens=True).to(device)
    rep = mdl_p(**toks).last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()
prot_vec = rep.astype(np.float32)  # ~480-D
print("[boot] Protein embedding:", prot_vec.shape)

def _embed_ligands(smiles_list, batch_size=64, max_length=256):
    vecs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        enc = tok_l(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = mdl_l(**enc).last_hidden_state
            cls = out[:, 0, :].detach().cpu().numpy().astype(np.float32)
        vecs.append(cls)
    return np.vstack(vecs) if vecs else np.zeros((0, mdl_l.config.hidden_size), dtype=np.float32)

# -----------------------------
# Try to load training data
# -----------------------------
df = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            if {'smiles','pAff'}.issubset(df.columns):
                print(f"[boot] Loaded dataset: {p} -> {df.shape}")
                break
        except Exception as e:
            print("[boot] Failed reading", p, e)

have_data = df is not None

# Placeholders initialized below
reg = None
clf = None
clf_cal = None
bins = None
q90_table = None
lig_tr = None
metrics_md = "*(Train a model or upload tem1_clean.csv to populate metrics here.)*"

def _train_models_from_df(df):
    global reg, clf, clf_cal, bins, q90_table, lig_tr, metrics_md

    df = df.dropna(subset=["smiles","pAff"]).reset_index(drop=True)

    # Ligand embeddings
    t0 = time.time()
    lig_X = _embed_ligands(df["smiles"].tolist())
    print(f"[train] Ligand embed {lig_X.shape} in {time.time()-t0:.1f}s")

    # Joint features with protein
    prot_X = np.repeat(prot_vec.reshape(1, -1), len(df), axis=0)
    X = np.hstack([prot_X, lig_X]).astype(np.float32)

    # Targets
    y = df["pAff"].astype(np.float32).values
    y_bin = (y >= PAFF_BINDER_THRESHOLD).astype(int)

    # Group-wise split by k-means clusters (scaffold-free)
    k = max(5, min(50, len(df)//50))
    km = KMeans(n_clusters=k, random_state=7, n_init=10)
    groups = km.fit_predict(lig_X)

    # custom split that holds out whole clusters
    def groupwise_split(groups, test_frac=0.2, seed=7):
        rng = np.random.default_rng(seed)
        keys = list(set(groups))
        rng.shuffle(keys)
        N = len(groups)
        target = int(N*test_frac)
        taken, test_idx = 0, []
        for key in keys:
            idx = np.where(groups==key)[0].tolist()
            test_idx.extend(idx)
            taken += len(idx)
            if taken >= target:
                break
        train_idx = sorted(set(range(N)) - set(test_idx))
        return np.array(train_idx), np.array(test_idx)

    tr_idx, te_idx = groupwise_split(groups, test_frac=0.2, seed=7)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    yb_tr, yb_te = y_bin[tr_idx], y_bin[te_idx]

    # Heads
    reg = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1
    ).fit(X_tr, y_tr)

    clf = LogisticRegression(max_iter=2000).fit(X_tr, yb_tr)

    # Metrics
    pred = reg.predict(X_te)
    try:
        rmse = mean_squared_error(y_te, pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_te, pred) ** 0.5
    r2 = r2_score(y_te, pred)
    p_bin = clf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(yb_te, p_bin)
    pr  = average_precision_score(yb_te, p_bin)

    # conditional q90 by predicted bin
    bins = np.linspace(float(pred.min()), float(pred.max()), 8)
    bin_idx = np.digitize(pred, bins)
    abs_err = np.abs(y_te - pred)
    q90_table = np.zeros(len(bins)+1, dtype=np.float32)
    for i in range(len(q90_table)):
        vals = abs_err[bin_idx==i]
        q90_table[i] = np.quantile(vals, 0.90) if len(vals)>0 else float(np.quantile(abs_err, 0.90))

    # calibration & similarity
    clf_cal = CalibratedClassifierCV(clf, method="isotonic", cv=3).fit(X_tr, yb_tr)
    lig_tr = lig_X[tr_idx]

    metrics_md = (
        f"**Eval (held-out)** — RMSE: {rmse:.2f} pAff (≈×{10**rmse:.1f}), "
        f"R²: {r2:.2f}, ROC-AUC: {roc:.2f}, PR-AUC: {pr:.2f}"
    )
    print("[train] done.")

def q90_for(p):
    i = int(np.digitize([p], bins)[0]) if bins is not None else 0
    i = max(0, min(i, len(q90_table)-1)) if q90_table is not None else 0
    return q90_table[i] if q90_table is not None else 0.75  # conservative fallback

# Try real training; otherwise install heuristic heads
if have_data:
    _train_models_from_df(df)
else:
    print("[boot] No dataset found — using heuristic heads (demo mode).")

    class HeuristicReg:
        def predict(self, X):
            # X: [B, Dp+Dl]; take ligand part and compute cosine to protein-projected vector
            Dp = prot_vec.shape[0]
            lig = X[:, Dp:]
            # project protein to ligand dims
            pv = prot_vec[:lig.shape[1]]
            pv = pv / (np.linalg.norm(pv) + 1e-8)
            lig_n = lig / (np.linalg.norm(lig, axis=1, keepdims=True)+1e-8)
            sim = (lig_n @ pv)
            return 5.5 + 2.0*(sim.clip(-1,1)+1)/2.0  # ~ [4.5,7.5]

    class HeuristicClf:
        def predict_proba(self, X):
            Dp = prot_vec.shape[0]
            lig = X[:, Dp:]
            pv = prot_vec[:lig.shape[1]]
            pv = pv / (np.linalg.norm(pv) + 1e-8)
            lig_n = lig / (np.linalg.norm(lig, axis=1, keepdims=True)+1e-8)
            sim = (lig_n @ pv)
            z = (sim - sim.min()) / (sim.max()-sim.min()+1e-8)
            p = 1/(1+np.exp(-4*(z-0.5)))
            return np.vstack([1-p, p]).T

    reg = HeuristicReg()
    clf = HeuristicClf()
    clf_cal = clf
    bins = np.linspace(4.0, 8.0, 8)
    q90_table = np.full(len(bins)+1, 0.75, dtype=np.float32)
    lig_tr = np.zeros((1, mdl_l.config.hidden_size), dtype=np.float32)
    metrics_md = "*(Demo mode — upload tem1_clean.csv to train real heads.)*"

# -----------------------------
# Prediction helpers
# -----------------------------
def train_similarity(smiles):
    enc = tok_l([smiles], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.inference_mode():
        lig = mdl_l(**enc).last_hidden_state[:,0,:].cpu().numpy().astype(np.float32)
    if lig_tr is None or lig_tr.shape[0]==0:
        return 0.0
    sim = cosine_similarity(lig, lig_tr)[0]
    return float(sim.max())

import matplotlib.pyplot as plt  # (already imported at top, fine to keep)

import traceback
import matplotlib.pyplot as plt  # keep after matplotlib.use("Agg")

def _blank_fig(width=3.6, height=0.6):
    fig = plt.figure(figsize=(width, height))
    plt.axis("off")
    return fig

def predict_smiles(smiles: str):
    try:
        # Empty input → friendly message + blank fig
        if not smiles:
            return "Please enter a SMILES", _blank_fig()

        # 1) ligand embedding
        enc = tok_l([smiles], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = mdl_l(**enc).last_hidden_state
            lig = out[:, 0, :].detach().cpu().numpy().astype(np.float32)

        # 2) joint feature
        fx = np.hstack([prot_vec.reshape(1, -1), lig]).astype(np.float32)

        # 3) regression + interval
        p_aff = float(reg.predict(fx)[0])
        q90   = q90_for(p_aff)
        p_lo, p_hi = p_aff - q90, p_aff + q90

        nM_center = pAff_to_nM(p_aff)
        nM_hi, nM_lo = pAff_to_nM(p_hi), pAff_to_nM(p_lo)

        # 4) calibrated binder probability
        try:
            p_cal = float(clf_cal.predict_proba(fx)[:, 1])
        except Exception:
            p_cal = float(clf.predict_proba(fx)[:, 1])
        label = conf_label(p_cal); mark = conf_emoji(p_cal)
        badge = " (≤1 µM)" if p_aff >= PAFF_BINDER_THRESHOLD else ""

        # 5) similarity
        sim = train_similarity(smiles)
        sim_note = (f"\nNearest-set similarity: {sim:.2f}"
                    if sim >= 0.60 else
                    f"\n⚠️ Low similarity to training set: {sim:.2f}")

        md = (
            f"**Predicted pAff:** {p_aff:.2f} (−log10 M){badge}  → **Kd ≈ {fmt_conc(nM_center)}**\n\n"
            f"**90% interval:** {p_lo:.2f} — {p_hi:.2f}  (≈ {fmt_conc(nM_hi)} to {fmt_conc(nM_lo)})\n\n"
            f"**Binder confidence:** {mark} {label} ({p_cal:.2f}){sim_note}\n"
        )

        # Mini bar to visualize P(binder)
        fig = plt.figure(figsize=(3.6, 0.6))
        ax = fig.add_axes([0.07, 0.35, 0.86, 0.35])
        ax.barh([0], [p_cal], height=0.6)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title("P(binder)")
        for spine in ax.spines.values():
            spine.set_visible(False)

        return md, fig

    except Exception as e:
        # Show the error inline so we can debug without checking logs
        tb = traceback.format_exc(limit=5)
        msg = f"❌ **Error:** {e}\n\n```\n{tb}\n```"
        return msg, _blank_fig()

def batch_predict(smiles_text):
    smi = _parse_smiles_block(smiles_text)
    if not smi:
        return [], np.array([]), np.array([])
    lig = _embed_ligands(smi)                              # (L, Dl)
    P   = np.repeat(prot_vec.reshape(1, -1), len(smi), 0)  # (L, Dp)
    X   = np.hstack([P, lig]).astype(np.float32)           # (L, Dp+Dl)
    p_aff  = reg.predict(X)
    p_bind = clf.predict_proba(X)[:, 1]
    return smi, p_aff, p_bind

def plot_paff_bars(names, paff, paff_thr=PAFF_BINDER_THRESHOLD):
    names = list(names); paff = np.array(paff, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(names)*0.6), 3.2))
    ax.bar(range(len(names)), paff)
    ax.axhline(paff_thr, linestyle="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:16]+("…" if len(n)>16 else "") for n in names], rotation=45, ha="right")
    ax.set_ylabel("Predicted pAff (−log10 M)"); ax.set_title("Batch predictions — pAff")
    plt.tight_layout()
    return fig

def plot_paff_vs_pbind(names, paff, pbind, hi=0.80, mid=0.60, paff_thr=PAFF_BINDER_THRESHOLD):
    names = list(names); paff = np.array(paff, dtype=float); pbind = np.array(pbind, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(paff, pbind, s=36)
    ax.axvline(paff_thr, linestyle="--"); ax.axhline(hi, linestyle="--"); ax.axhline(mid, linestyle="--")
    top = np.argsort(-(paff + pbind))[:10]
    for i in top:
        lbl = names[i][:18] + ("…" if len(names[i]) > 18 else "")
        ax.annotate(lbl, (paff[i], pbind[i]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Predicted pAff (−log10 M)"); ax.set_ylabel("Calibrated P(binder)")
    ax.set_title("Batch predictions"); plt.tight_layout()
    return fig

def heatmap_predict(smiles_block):
    smi_list = _parse_smiles_block(smiles_block)
    if not smi_list:
        fig = plt.figure(figsize=(4, 2))
        plt.axis("off")
        plt.text(0.5, 0.5, "No SMILES provided", ha="center", va="center")
        return fig

    # Embed ligands
    ligs = _embed_ligands(smi_list)
    # Joint features (protein + ligands)
    pv_rep = np.repeat(prot_vec.reshape(1, -1), len(smi_list), axis=0)
    fx = np.hstack([pv_rep, ligs]).astype(np.float32)

    # Predict pAff (single protein row)
    p_affs = reg.predict(fx)  # shape (L,)
    M = p_affs.reshape(1, -1)  # 1 x L

    fig, ax = plt.subplots(figsize=(max(6, len(smi_list)*0.8), 2.8))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(range(len(smi_list)))
    ax.set_xticklabels([s[:14] + ("…" if len(s) > 14 else "") for s in smi_list],
                       rotation=45, ha="right")
    ax.set_yticks([0]); ax.set_yticklabels(["TEM-1 (WT)"])
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Predicted pAff")

    # Mark predicted binders (>= threshold)
    for j in range(M.shape[1]):
        if M[0, j] >= PAFF_BINDER_THRESHOLD:
            ax.text(j, 0, "★", ha="center", va="center", color="white", fontsize=12)

    ax.set_title("Heatmap — predicted pAff (higher is better)")
    plt.tight_layout()
    return fig


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Antibiotic Resistance Target Finder — TEM-1") as demo:
    gr.Markdown("""\
# Antibiotic Resistance Target Finder — TEM-1
**Goal:** Predict how tightly a small molecule binds **TEM-1 β-lactamase** variants.
**How to use (2 steps):**
1) Paste a **SMILES** string and click **Submit** to get a prediction.
2) (Optional) Paste multiple SMILES for batch plots and a heatmap.
*Protein embeddings:* ESM-2 (35M) · *Ligand embeddings:* ChemBERTa · *Models:* XGBoost + LogisticRegression
""")

    with gr.Row():
        smi_in = gr.Textbox(label="SMILES", placeholder="e.g., CC1=CC(=O)C=CC1=O", lines=1)
        btn = gr.Button("Submit", variant="primary")

    out_md = gr.Markdown()
    out_plot = gr.Plot()

    btn.click(fn=predict_smiles, inputs=smi_in, outputs=[out_md, out_plot])

    gr.Markdown("""---
### Batch mode (paste 1–100 SMILES separated by newlines, commas, or semicolons)
""")

    smi_batch = gr.Textbox(label="Batch SMILES", lines=6, placeholder="SMILES per line ...")
    
    with gr.Row():
        btn_bars = gr.Button("Bar chart (pAff)")
        btn_scatter = gr.Button("Scatter (pAff vs P(binder))")
        btn_heat = gr.Button("Heatmap")
    
    plot1 = gr.Plot()
    plot2 = gr.Plot()
    plot3 = gr.Plot()
    
    def _bars(smiblock):
        names, paff, pbind = batch_predict(smiblock)
        return plot_paff_bars(names, paff)
    
    def _scatter(smiblock):
        names, paff, pbind = batch_predict(smiblock)
        return plot_paff_vs_pbind(names, paff, pbind)
    
    def _heat(smiblock):
        return heatmap_predict(smiblock)
    
    btn_bars.click(_bars, inputs=smi_batch, outputs=plot1)
    btn_scatter.click(_scatter, inputs=smi_batch, outputs=plot2)
    btn_heat.click(_heat, inputs=smi_batch, outputs=plot3)


    with gr.Accordion("Model card: assumptions, metrics & limits", open=False):
        gr.Markdown("""\
**Compute footprint:** small (≤50M embeddings + lightweight heads). Runs on CPU in Spaces.  
%s
**Assumptions / caveats**
- Trained on **TEM-1** datasets; predictions for very dissimilar chemotypes are less certain.
- Reported “confidence” is **calibrated** on a held-out set; not a substitute for wet-lab validation.
- Use as a **ranking/triage** tool, not as a definitive activity claim.
**pAff** is −log10(Kd in molar). Bigger is better. Example: 1 µM → pAff=6; 100 nM → 7; 10 nM → 8.
""" % metrics_md)

demo.launch()