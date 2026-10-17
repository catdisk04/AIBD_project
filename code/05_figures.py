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
