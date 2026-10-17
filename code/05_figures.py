"""
Step 5: combined ROC and precision-recall figure for both modalities, from the
first-repeat out-of-fold predictions saved by 02 and 03.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from common import TAB, savefig

snp = pd.read_csv(TAB / "oof_snp.csv")
curves = [
    ("SNP: Age + Sex (LR)", snp.y, snp["Age + Sex (LR)"], "#aaaaaa", "--"),
    ("SNP: Additive (LR)", snp.y, snp["Additive + Age + Sex (LR)"], "#74add1", "-"),
    ("SNP: Dominant (LR)", snp.y, snp["Dominant + Age + Sex (LR)"], "#2166ac", "-"),
    ("SNP: Additive (XGBoost)", snp.y, snp["Additive + Age + Sex (XGBoost)"], "#F4A261", "-."),
]
for name, kind, c, ls in [("GEX: L1-LR", "L1-LR", "#E05C5C", "-"),
                          ("GEX: XGBoost", "XGBoost", "#8B1A1A", "-."),
                          ("GEX: batch label only", "Batch only", "#555555", ":")]:
    g = pd.read_csv(TAB / f"oof_gex_P3_{kind}.csv")
    curves.append((name, g.y, g.prob, c, ls))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for name, y, p, c, ls in curves:
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, color=c, ls=ls, lw=1.8, label=f"{name} ({roc_auc_score(y, p):.3f})")
    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr, color=c, ls=ls, lw=1.8, label=f"{name} ({average_precision_score(y, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8, alpha=.5)
ax[1].axhline(snp.y.mean(), color="#2166ac", lw=.8, ls=":", alpha=.7)
ax[1].axhline(g.y.mean(), color="#E05C5C", lw=.8, ls=":", alpha=.7)
ax[1].text(0.01, snp.y.mean() + .01, "SNP prevalence", fontsize=7, color="#2166ac")
ax[1].text(0.01, g.y.mean() + .01, "GEX prevalence", fontsize=7, color="#E05C5C")
ax[0].set(xlabel="False positive rate", ylabel="True positive rate")
ax[1].set(xlabel="Recall", ylabel="Precision", ylim=(0.3, 1.02))
ax[0].set_title("A  ROC (out-of-fold, ROC-AUC)", loc="left", fontweight="bold")
ax[1].set_title("B  Precision-recall (out-of-fold, PR-AUC)", loc="left", fontweight="bold")
for a in ax:
    a.legend(fontsize=7, loc="lower right" if a is ax[0] else "lower left", framealpha=.9)
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
savefig(fig, "fig_roc_pr")


# ------------------------------------------------------------------ SHAP summary figure
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CMAP = LinearSegmentedColormap.from_list("lowhigh", ["#2166ac", "#d9d9d9", "#b2182b"])


def beeswarm(ax, shap_df, feat_df, order, labels, title):
    rng = np.random.default_rng(0)
    for i, f in enumerate(order[::-1]):
        sv = shap_df[f].values
        fv = feat_df[f].values
        # colour: feature value scaled to [0, 1] between its 5th and 95th percentiles
        # (genotypes 0/1/2 map to low / middle / high)
        lo, hi = np.percentile(fv, [5, 95])
        if hi == lo:
            lo, hi = fv.min(), fv.max()
        c = np.clip((fv - lo) / (hi - lo), 0, 1)
        jitter = rng.uniform(-0.32, 0.32, len(sv))
        ax.scatter(sv, i + jitter, c=c, cmap=CMAP, s=5, alpha=.7, linewidths=0, vmin=0, vmax=1)
    ax.axvline(0, color="k", lw=.6)
    ax.set_yticks(range(len(order)), [labels.get(f, f) for f in order[::-1]], fontsize=8)
    ax.set_xlabel("SHAP value (impact on predicted SLE log-odds)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


snp_shap = pd.read_csv(TAB / "snp_shap_oof.csv")
snp_feat = pd.read_csv(TAB / "snp_shap_oof_features.csv")
snp_rank = pd.read_csv(TAB / "snp_shap.csv", index_col=0).index[:12].tolist()
gex_shap = pd.read_csv(TAB / "gex_shap_oof.csv")
gex_feat = pd.read_csv(TAB / "gex_shap_oof_features.csv")
gex_rank = pd.read_csv(TAB / "gex_explanations.csv", index_col=0).index.tolist()
de = pd.read_csv(TAB / "de_batch_adjusted.csv", index_col=0)
gex_labels = {g: (f"{g} *" if de.loc[g, "q (BH)"] < 0.05 else g) for g in gex_rank}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
snp_labels = {"AGE": "Age", "sex": "Sex", "TIRAPS180L* red filled rep smp": "TIRAP S180L",
              "stat4 new 7574865rs": "STAT4 rs7574865"}
beeswarm(ax[0], snp_shap, snp_feat, snp_rank, snp_labels,
         "A  SNP model (XGBoost, additive + age + sex)")
beeswarm(ax[1], gex_shap, gex_feat, gex_rank, gex_labels,
         "B  Expression model (XGBoost, leakage-free)")
sm_ = plt.cm.ScalarMappable(cmap=CMAP)
cb = fig.colorbar(sm_, ax=ax, fraction=.025, pad=.02, ticks=[0, 1])
cb.ax.set_yticklabels(["low", "high"], fontsize=8)
cb.set_label("Feature value", fontsize=8)
savefig(fig, "fig_shap")
print("done")
