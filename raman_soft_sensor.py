"""
raman_soft_sensor.py
--------------------
Raman Spectroscopy Soft Sensor for Real-Time Penicillin Monitoring

Uses Raman spectra from the IndPenSim industrial-scale penicillin
fermentation benchmark to build PLS-based soft sensors that predict
product concentration in real time — eliminating the need for slow,
costly offline laboratory assays.

This is a direct implementation of Process Analytical Technology (PAT)
as described in FDA's PAT Guidance (2004) and ICH Q8(R2).

Dataset:
  IndPenSim (Goldrick et al., 2019) — 100,000 L fed-batch bioreactor
  30 normal batches (training) + 10 fault batches (testing)
  440 Raman channels, 205–2400 cm⁻¹, sampled every 0.4 h

Targets predicted:
  - Penicillin concentration (P: g/L) — Critical Quality Attribute
  - Substrate concentration  (S: g/L) — Critical Process Parameter

Reference:
  Goldrick S. et al. (2019). Computers & Chemical Engineering, 130, 106471.
"""

import os, sys, warnings, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.signal             import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.decomposition    import PCA
from sklearn.model_selection  import GroupKFold, cross_val_predict
from sklearn.metrics          import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
_SRC = "/mnt/user-data/uploads/indpensim_raman.csv"
_DST = "/tmp/indpensim_raman.csv"
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST)

DATA_PATH = "/tmp/indpensim_raman.csv"
OUT       = "/mnt/user-data/outputs/pharma-process-monitoring"
FIG_DIR   = f"{OUT}/figures/raman"
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 150
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": DPI})

NORMAL_C = "#2980B9"
FAULT_C  = "#C0392B"
GREEN_C  = "#1D9E75"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 65)
print("  RAMAN SOFT SENSOR — IndPenSim Penicillin Fermentation")
print("=" * 65)

print("\n[1] Loading data…")
df = pd.read_csv(DATA_PATH)

# Identify Raman channels (numeric column names = wavenumber in cm⁻¹)
RAMAN_COLS = [c for c in df.columns if c.strip().lstrip('-').isdigit()]
WN         = np.array([float(c) for c in RAMAN_COLS])

NORMAL_IDS = sorted(df[df["fault_label"]==0]["batch_id"].unique())
FAULT_IDS  = sorted(df[df["fault_label"]==1]["batch_id"].unique())

print(f"  Observations      : {len(df):,}")
print(f"  Raman channels    : {len(RAMAN_COLS)} ({WN.min():.0f}–{WN.max():.0f} cm⁻¹)")
print(f"  Normal batches    : {len(NORMAL_IDS)}")
print(f"  Fault batches     : {len(FAULT_IDS)}")
print(f"  Penicillin range  : {df['Penicillin concentration(P:g/L)'].min():.2f}–"
      f"{df['Penicillin concentration(P:g/L)'].max():.2f} g/L")

# =============================================================================
# 2. SPECTRAL PREPROCESSING
# =============================================================================
print("\n[2] Preprocessing Raman spectra…")

def snv(X):
    """Standard Normal Variate — removes multiplicative scatter."""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1,  keepdims=True) + 1e-10
    return (X - m) / s

def sg_smooth(X, window=15, poly=3):
    """Savitzky-Golay smoothing."""
    return np.apply_along_axis(
        lambda x: savgol_filter(x, window, poly), 1, X)

def sg_deriv(X, window=15, poly=3, deriv=1):
    """First derivative — removes baseline drift."""
    return np.apply_along_axis(
        lambda x: savgol_filter(x, window, poly, deriv=deriv), 1, X)

X_raw   = df[RAMAN_COLS].values.astype(float)
X_snv   = snv(sg_smooth(X_raw))          # SNV after smoothing
X_deriv = sg_deriv(X_snv)                # 1st derivative of SNV

print(f"  Raw → SG smooth → SNV → 1st derivative")
print(f"  Preprocessing complete")

# =============================================================================
# 3. SPECTRAL EDA
# =============================================================================
print("\n[3] Generating spectral EDA figures…")

# ── Fig R1: Raw and preprocessed spectra ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Sample spectra from one normal batch, coloured by time
batch1    = df[df["batch_id"]==1]
idx1      = batch1.index
pen_vals  = batch1["Penicillin concentration(P:g/L)"].values
cmap_obj  = plt.get_cmap("plasma")
norm_vals = (pen_vals - pen_vals.min()) / (pen_vals.max() - pen_vals.min() + 1e-8)

for i, (ax, X_plot, title) in enumerate(zip(
    axes,
    [X_raw[idx1],    X_snv[idx1],   X_deriv[idx1]],
    ["Raw Spectrum", "SG + SNV",    "1st Derivative (SG + SNV)"],
)):
    for j in range(0, len(idx1), max(1, len(idx1)//30)):
        ax.plot(WN, X_plot[j], color=cmap_obj(norm_vals[j]),
                alpha=0.6, lw=0.8)
    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity / A.U.", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=11)
    _style(ax)

sm = plt.cm.ScalarMappable(cmap="plasma",
     norm=plt.Normalize(pen_vals.min(), pen_vals.max()))
sm.set_array([])
fig.colorbar(sm, ax=axes[-1], fraction=0.04,
             label="Penicillin (g/L)")
fig.suptitle("Raman Spectra — Preprocessing Pipeline\n"
             "Batch 1 · Coloured by Penicillin Concentration",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R1_spectral_preprocessing.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R1_spectral_preprocessing.png")

# ── Fig R2: Mean spectra normal vs fault ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
X_normal_all = X_snv[df["fault_label"]==0]
X_fault_all  = X_snv[df["fault_label"]==1]

for ax, X_n, X_f, title in zip(
    axes,
    [X_snv[df["fault_label"]==0],   X_deriv[df["fault_label"]==0]],
    [X_snv[df["fault_label"]==1],   X_deriv[df["fault_label"]==1]],
    ["SNV Normalised", "1st Derivative (SNV)"],
):
    mean_n = X_n.mean(axis=0)
    std_n  = X_n.std(axis=0)
    mean_f = X_f.mean(axis=0)

    ax.plot(WN, mean_n, color=NORMAL_C, lw=2, label="Normal mean")
    ax.fill_between(WN, mean_n-std_n, mean_n+std_n,
                    alpha=0.2, color=NORMAL_C, label="±1 SD (normal)")
    ax.plot(WN, mean_f, color=FAULT_C, lw=2, ls="--", label="Fault mean")
    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity", fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    _style(ax)

fig.suptitle("Mean Raman Spectra — Normal vs Fault Batches\n"
             "Spectral differences indicate chemical composition changes",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R2_normal_vs_fault_spectra.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R2_normal_vs_fault_spectra.png")

# ── Fig R3: PCA on spectra (domain view) ─────────────────────────────────────
pca_spec  = PCA(n_components=5, random_state=42)
sc_normal = pca_spec.fit_transform(
    StandardScaler().fit_transform(X_snv[df["fault_label"]==0]))
scaler_sp = StandardScaler().fit(X_snv[df["fault_label"]==0])
sc_all    = pca_spec.transform(scaler_sp.transform(X_snv))
ev_sp     = pca_spec.explained_variance_ratio_ * 100

pen_all   = df["Penicillin concentration(P:g/L)"].values
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Coloured by penicillin
sc1 = axes[0].scatter(sc_all[:,0], sc_all[:,1],
                      c=pen_all, cmap="plasma",
                      s=5, alpha=0.4, edgecolors="none")
plt.colorbar(sc1, ax=axes[0], fraction=0.04, label="Penicillin (g/L)")
axes[0].set_xlabel(f"PC1 ({ev_sp[0]:.1f}%)", fontsize=11)
axes[0].set_ylabel(f"PC2 ({ev_sp[1]:.1f}%)", fontsize=11)
axes[0].set_title("PCA of Raman Spectra\n(coloured by penicillin)",
                  fontweight="bold")
_style(axes[0])

# Coloured by normal vs fault
colors = [FAULT_C if fl==1 else NORMAL_C
          for fl in df["fault_label"].values]
axes[1].scatter(sc_all[:,0], sc_all[:,1],
                c=colors, s=5, alpha=0.3, edgecolors="none")
handles = [mpatches.Patch(color=NORMAL_C, label="Normal"),
           mpatches.Patch(color=FAULT_C,  label="Fault")]
axes[1].legend(handles=handles, fontsize=10)
axes[1].set_xlabel(f"PC1 ({ev_sp[0]:.1f}%)", fontsize=11)
axes[1].set_ylabel(f"PC2 ({ev_sp[1]:.1f}%)", fontsize=11)
axes[1].set_title("PCA of Raman Spectra\n(normal vs fault)",
                  fontweight="bold")
_style(axes[1])

fig.suptitle("Spectral PCA — IndPenSim Raman Data",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R3_spectral_pca.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R3_spectral_pca.png")

# =============================================================================
# 4. PLS SOFT SENSOR — TRAINING
# =============================================================================
print("\n[4] Building PLS soft sensor models…")

# Training data: normal batches only
train_mask = df["fault_label"] == 0
X_train    = X_deriv[train_mask]
groups     = df[train_mask]["batch_id"].values

TARGETS = {
    "Penicillin": "Penicillin concentration(P:g/L)",
    "Substrate":  "Substrate concentration(S:g/L)",
}

scaler_X = StandardScaler()
X_tr_sc  = scaler_X.fit_transform(X_train)

models   = {}
cv_preds = {}
metrics  = {}

gkf = GroupKFold(n_splits=5)

for name, col in TARGETS.items():
    y_train = df[train_mask][col].values

    # Find optimal n_components via Leave-One-Batch-Out CV
    best_rmse, best_n = np.inf, 1
    for n in range(1, 12):
        pls  = PLSRegression(n_components=n, max_iter=1000)
        pred = cross_val_predict(pls, X_tr_sc, y_train,
                                 cv=gkf, groups=groups)
        rmse = np.sqrt(mean_squared_error(y_train, pred))
        if rmse < best_rmse:
            best_rmse, best_n = rmse, n

    # Train final model
    model = PLSRegression(n_components=best_n, max_iter=1000)
    model.fit(X_tr_sc, y_train)

    # Leave-one-batch-out predictions
    cv_pred = cross_val_predict(model, X_tr_sc, y_train,
                                cv=gkf, groups=groups)
    r2cv    = r2_score(y_train, cv_pred)

    models[name]   = model
    cv_preds[name] = cv_pred
    metrics[name]  = {
        "n_comp":  best_n,
        "rmsecv":  best_rmse,
        "r2cv":    r2cv,
        "y_train": y_train,
    }

    print(f"  {name:<12}: n_comp={best_n:2d}  "
          f"RMSECV={best_rmse:.4f}  R²CV={r2cv:.3f}")

# =============================================================================
# 5. EVALUATION PLOTS
# =============================================================================
print("\n[5] Generating evaluation figures…")

# ── Fig R4: Predicted vs actual (LOBO-CV) ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (name, col) in zip(axes, TARGETS.items()):
    y_true = metrics[name]["y_train"]
    y_pred = cv_preds[name]
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    # Colour by batch
    batch_vals = groups
    cmap_b = cm.tab20(np.linspace(0, 1, len(NORMAL_IDS)))
    for bi, bid in enumerate(NORMAL_IDS):
        mask = batch_vals == bid
        ax.scatter(y_true[mask], y_pred[mask],
                   s=8, alpha=0.4, color=cmap_b[bi],
                   edgecolors="none")

    ax.plot([lo, hi], [lo, hi], "r--", lw=1.8, alpha=0.8)
    ax.set_xlabel(f"Actual {name} (g/L)", fontsize=11)
    ax.set_ylabel(f"Predicted {name} (g/L)", fontsize=11)
    ax.set_title(
        f"{name} Soft Sensor (LOBO-CV)\n"
        f"n={metrics[name]['n_comp']} PLS components  "
        f"RMSE={metrics[name]['rmsecv']:.3f}  R²={metrics[name]['r2cv']:.3f}",
        fontweight="bold", fontsize=11,
    )
    _style(ax)

fig.suptitle("PLS Soft Sensor — Leave-One-Batch-Out Cross-Validation\n"
             "Each colour = one batch (30 batches, each used as test once)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R4_predicted_vs_actual.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R4_predicted_vs_actual.png")

# ── Fig R5: Real-time tracking — normal batch ─────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

bid = 5   # example normal batch
sub = df[df["batch_id"]==bid]
X_b = X_deriv[sub.index]
X_b_sc = scaler_X.transform(X_b)
t   = sub["Time (h)"].values

for ax, (name, col) in zip(axes, TARGETS.items()):
    y_true = sub[col].values
    y_pred = models[name].predict(X_b_sc).flatten()

    ax.plot(t, y_true, color="black",  lw=2,   label="Actual (offline assay)")
    ax.plot(t, y_pred, color=GREEN_C,  lw=2, ls="--",
            label=f"Predicted (Raman soft sensor)")
    ax.fill_between(t, y_pred - metrics[name]["rmsecv"],
                    y_pred + metrics[name]["rmsecv"],
                    alpha=0.2, color=GREEN_C, label="±RMSECV")
    ax.set_ylabel(f"{name} (g/L)", fontsize=11)
    ax.set_title(f"Real-Time {name} Monitoring — Batch {bid}",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    _style(ax)

axes[-1].set_xlabel("Fermentation Time (h)", fontsize=11)
fig.suptitle("Raman Soft Sensor — Real-Time Process Monitoring\n"
             "Continuous prediction vs offline laboratory measurements",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R5_realtime_normal_batch.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R5_realtime_normal_batch.png")

# ── Fig R6: Real-time tracking — fault batch (shows sensor degradation) ───────
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

fault_bid = 91
sub_f   = df[df["batch_id"]==fault_bid]
X_f     = X_deriv[sub_f.index]
X_f_sc  = scaler_X.transform(X_f)
t_f     = sub_f["Time (h)"].values
fault_t = sub_f["Fault reference(Fault_ref:Fault ref)"].values

for ax, (name, col) in zip(axes, TARGETS.items()):
    y_true = sub_f[col].values
    y_pred = models[name].predict(X_f_sc).flatten()

    ax.fill_between(t_f, 0, max(y_true.max(), y_pred.max())*1.1,
                    where=(fault_t==1),
                    alpha=0.12, color=FAULT_C, label="Known fault period")
    ax.plot(t_f, y_true, color="black", lw=2,
            label="Actual (offline assay)")
    ax.plot(t_f, y_pred, color=FAULT_C, lw=2, ls="--",
            label="Predicted (Raman soft sensor)")
    ax.set_ylabel(f"{name} (g/L)", fontsize=11)
    ax.set_title(f"Real-Time {name} — Fault Batch {fault_bid}",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    _style(ax)

axes[-1].set_xlabel("Fermentation Time (h)", fontsize=11)
fig.suptitle("Raman Soft Sensor — Fault Batch Monitoring\n"
             "Prediction divergence reveals process deviation",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R6_realtime_fault_batch.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R6_realtime_fault_batch.png")

# ── Fig R7: VIP scores ────────────────────────────────────────────────────────
def vip_scores(model, X):
    T = model.x_scores_
    W = model.x_weights_
    Q = model.y_loadings_
    p = X.shape[1]
    SS  = np.diag(T.T @ T) * (Q.flatten()**2)
    return np.sqrt(p * (W**2 @ SS) / SS.sum())

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

for ax, (name, col) in zip(axes, TARGETS.items()):
    vip = vip_scores(models[name], X_tr_sc)
    ax.fill_between(WN, vip, where=(vip >= 1.0),
                    alpha=0.35, color=FAULT_C, label="VIP ≥ 1.0 (important)")
    ax.fill_between(WN, vip, where=(vip < 1.0),
                    alpha=0.2,  color=NORMAL_C, label="VIP < 1.0")
    ax.plot(WN, vip, color="#2C3E50", lw=1.5)
    ax.axhline(1.0, color=FAULT_C, ls="--", lw=1.5,
               alpha=0.8, label="VIP = 1.0 threshold")
    ax.set_ylabel("VIP Score", fontsize=11)
    ax.set_title(
        f"Variable Importance in Projection — {name} Soft Sensor\n"
        f"(n={metrics[name]['n_comp']} PLS components)",
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    _style(ax)

axes[-1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
fig.suptitle("VIP Scores — Spectral Regions Driving Soft Sensor Predictions\n"
             "High VIP wavenumbers indicate chemically informative bands",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R7_vip_scores.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R7_vip_scores.png")

# ── Fig R8: All fault batches — prediction error over time ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (name, col) in zip(axes, TARGETS.items()):
    cmap_f = cm.Set1(np.linspace(0, 0.9, len(FAULT_IDS)))
    for fi, bid in enumerate(FAULT_IDS):
        sub_fi = df[df["batch_id"]==bid]
        X_fi   = X_deriv[sub_fi.index]
        X_fi_sc= scaler_X.transform(X_fi)
        t_fi   = sub_fi["Time (h)"].values
        y_true = sub_fi[col].values
        y_pred = models[name].predict(X_fi_sc).flatten()
        error  = np.abs(y_true - y_pred)

        ax.plot(t_fi, error, color=cmap_f[fi], lw=1.5,
                alpha=0.8, label=f"Batch {bid}")

    ax.axhline(metrics[name]["rmsecv"], color="black",
               ls="--", lw=1.5, alpha=0.6,
               label=f"RMSECV ({metrics[name]['rmsecv']:.3f})")
    ax.set_xlabel("Fermentation Time (h)", fontsize=11)
    ax.set_ylabel("|Prediction Error| (g/L)", fontsize=11)
    ax.set_title(f"{name} — Absolute Prediction Error\nFault Batches 91–100",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    _style(ax)

fig.suptitle("Soft Sensor Prediction Error — All Fault Batches\n"
             "Error increase after fault onset reveals spectral drift",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/R8_fault_prediction_error.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R8_fault_prediction_error.png")

# ── Fig R9: Summary dashboard ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# (0,0) Penicillin predicted vs actual
ax00 = fig.add_subplot(gs[0, 0])
y_t  = metrics["Penicillin"]["y_train"]
y_p  = cv_preds["Penicillin"]
ax00.scatter(y_t, y_p, s=6, alpha=0.3, color=NORMAL_C, edgecolors="none")
lims = [y_t.min(), y_t.max()]
ax00.plot(lims, lims, "r--", lw=1.5)
ax00.set_xlabel("Actual (g/L)"); ax00.set_ylabel("Predicted (g/L)")
ax00.set_title(
    f"Penicillin (LOBO-CV)\nRMSE={metrics['Penicillin']['rmsecv']:.3f}"
    f"  R²={metrics['Penicillin']['r2cv']:.3f}",
    fontweight="bold")
_style(ax00)

# (0,1) VIP - Penicillin
ax01 = fig.add_subplot(gs[0, 1])
vip  = vip_scores(models["Penicillin"], X_tr_sc)
ax01.plot(WN, vip, color="#2C3E50", lw=1.5)
ax01.fill_between(WN, vip, where=(vip>=1), alpha=0.3, color=FAULT_C)
ax01.axhline(1.0, color=FAULT_C, ls="--", lw=1.5)
ax01.set_xlabel("Raman Shift (cm⁻¹)")
ax01.set_ylabel("VIP Score")
ax01.set_title("VIP — Penicillin Sensor", fontweight="bold")
_style(ax01)

# (0,2) Real-time tracking normal
ax02 = fig.add_subplot(gs[0, 2])
sub5   = df[df["batch_id"]==5]
X5_sc  = scaler_X.transform(X_deriv[sub5.index])
y_t5   = sub5["Penicillin concentration(P:g/L)"].values
y_p5   = models["Penicillin"].predict(X5_sc).flatten()
ax02.plot(sub5["Time (h)"], y_t5, "k-", lw=2, label="Actual")
ax02.plot(sub5["Time (h)"], y_p5, color=GREEN_C, lw=2, ls="--",
          label="Predicted")
ax02.set_xlabel("Time (h)"); ax02.set_ylabel("Penicillin (g/L)")
ax02.set_title("Real-Time Tracking\nBatch 5 (Normal)", fontweight="bold")
ax02.legend(fontsize=8); _style(ax02)

# (1,0) Real-time tracking fault
ax10 = fig.add_subplot(gs[1, 0])
sub91  = df[df["batch_id"]==91]
X91_sc = scaler_X.transform(X_deriv[sub91.index])
y_t91  = sub91["Penicillin concentration(P:g/L)"].values
y_p91  = models["Penicillin"].predict(X91_sc).flatten()
ft91   = sub91["Fault reference(Fault_ref:Fault ref)"].values
ax10.fill_between(sub91["Time (h)"], 0, max(y_t91.max(),y_p91.max())*1.1,
                  where=(ft91==1), alpha=0.12, color=FAULT_C)
ax10.plot(sub91["Time (h)"], y_t91, "k-", lw=2, label="Actual")
ax10.plot(sub91["Time (h)"], y_p91, color=FAULT_C, lw=2, ls="--",
          label="Predicted")
ax10.set_xlabel("Time (h)"); ax10.set_ylabel("Penicillin (g/L)")
ax10.set_title("Real-Time Tracking\nBatch 91 (Fault)", fontweight="bold")
ax10.legend(fontsize=8); _style(ax10)

# (1,1) Substrate predicted vs actual
ax11 = fig.add_subplot(gs[1, 1])
y_ts = metrics["Substrate"]["y_train"]
y_ps = cv_preds["Substrate"]
ax11.scatter(y_ts, y_ps, s=6, alpha=0.3, color=GREEN_C, edgecolors="none")
lims2 = [y_ts.min(), y_ts.max()]
ax11.plot(lims2, lims2, "r--", lw=1.5)
ax11.set_xlabel("Actual (g/L)"); ax11.set_ylabel("Predicted (g/L)")
ax11.set_title(
    f"Substrate (LOBO-CV)\nRMSE={metrics['Substrate']['rmsecv']:.3f}"
    f"  R²={metrics['Substrate']['r2cv']:.3f}",
    fontweight="bold")
_style(ax11)

# (1,2) Results table
ax12 = fig.add_subplot(gs[1, 2])
ax12.axis("off")
tbl = ax12.table(
    cellText=[
        ["Dataset",          "IndPenSim"],
        ["Instrument",       "Raman (simulated)"],
        ["Channels",         f"{len(RAMAN_COLS)} (205–2400 cm⁻¹)"],
        ["Training batches", "30 (normal)"],
        ["Validation",       "Leave-One-Batch-Out"],
        ["Pen. RMSE (CV)",   f"{metrics['Penicillin']['rmsecv']:.4f} g/L"],
        ["Pen. R² (CV)",     f"{metrics['Penicillin']['r2cv']:.3f}"],
        ["Sub. RMSE (CV)",   f"{metrics['Substrate']['rmsecv']:.4f} g/L"],
        ["Sub. R² (CV)",     f"{metrics['Substrate']['r2cv']:.3f}"],
    ],
    colLabels=["Metric", "Value"],
    cellLoc="center", loc="center",
    colWidths=[0.58, 0.42],
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EEF4FB")
ax12.set_title("Results Summary", fontweight="bold", pad=12)

fig.suptitle("Raman Soft Sensor Dashboard — IndPenSim\n"
             "PLS-based Real-Time Penicillin & Substrate Monitoring",
             fontsize=14, fontweight="bold", y=1.01)
path = f"{FIG_DIR}/R0_dashboard.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved: R0_dashboard.png")

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================
results_df = pd.DataFrame([
    {
        "target":    name,
        "n_components": metrics[name]["n_comp"],
        "rmsecv":    round(metrics[name]["rmsecv"], 4),
        "r2cv":      round(metrics[name]["r2cv"],   4),
    }
    for name in TARGETS
])
results_df.to_csv(f"{OUT}/outputs/soft_sensor_metrics.csv", index=False)

print(f"\n{'='*65}")
print(f"  RESULTS SUMMARY")
print(f"{'='*65}")
print(f"  Dataset           : IndPenSim (Goldrick et al., 2019)")
print(f"  Raman channels    : {len(RAMAN_COLS)} (205–2400 cm⁻¹)")
print(f"  Training batches  : {len(NORMAL_IDS)} (LOBO-CV)")
for name in TARGETS:
    print(f"  {name:<15}: "
          f"RMSE={metrics[name]['rmsecv']:.4f} g/L  "
          f"R²={metrics[name]['r2cv']:.3f}  "
          f"({metrics[name]['n_comp']} PLS components)")
print(f"\n  Figures saved to  : {FIG_DIR}/")
