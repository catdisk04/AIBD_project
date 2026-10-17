"""
Step 1: scale harmonisation, gene filtering and ComBat batch correction on the full
cohort, plus the before/after diagnostics reported in the paper.

This is the descriptive analysis (PCA, batch R^2, silhouettes, Cohen's d). The
classifiers in 02_gex_classification.py do NOT use the matrix produced here; they
refit ComBat inside every training fold.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from common import (load_expression, harmonize_scale, combat_ok_genes,
                    disease_covariates, savefig, TAB, RESULTS)
from combat import ComBat

df = harmonize_scale(load_expression())
genes = combat_ok_genes(df)
print("Genes kept:", genes)

# batch composition table
comp = pd.crosstab(df["batch"], df["Disease_State"])
comp.insert(0, "Samples", comp.sum(1))
comp["Subjects"] = df.groupby("batch")["subject"].nunique()
comp.to_csv(TAB / "batch_composition.csv")
print(comp)

X = df[genes].values
batch = df["batch"].values
disease = df["Disease_State"].values
covar = disease_covariates(disease)

X_mo = ComBat(mean_only=True).fit_transform(X, batch, covar)
X_full = ComBat(mean_only=False).fit_transform(X, batch, covar)
X_nocov = ComBat(mean_only=True).fit_transform(X, batch, None)   # sensitivity: no labels used


def r2_by_group(v, groups):
    v = np.asarray(v, float)
    s = pd.Series(v).groupby(np.asarray(groups))
    return (s.size() * (s.mean() - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum()


def zscore(A):
    return (A - A.mean(0)) / A.std(0)


usable = [b for b in pd.unique(batch)
          if ((batch == b) & (disease == "SLE")).sum() >= 3
          and ((batch == b) & (disease == "Healthy Control")).sum() >= 3]


def cohens_d(A):
    out = pd.DataFrame(index=genes, columns=usable, dtype=float)
    for b in usable:
        s = A[(batch == b) & (disease == "SLE")]
        h = A[(batch == b) & (disease == "Healthy Control")]
        n1, n2 = len(s), len(h)
        sp = np.sqrt(((n1 - 1) * s.var(0, ddof=1) + (n2 - 1) * h.var(0, ddof=1)) / (n1 + n2 - 2))
        out[b] = (s.mean(0) - h.mean(0)) / sp
    return out


def evaluate(A):
    Az = zscore(A)
    pcs = PCA(2, random_state=0).fit_transform(Az)
    d = cohens_d(A)
    sign = np.sign(d)
    concord = (sign.T == np.sign(sign.sum(1))).T.mean(1).mean()
    return {
        "Mean batch R2 (genes)": np.mean([r2_by_group(A[:, i], batch) for i in range(A.shape[1])]),
        "Mean disease R2 (genes)": np.mean([r2_by_group(A[:, i], disease) for i in range(A.shape[1])]),
        "PC1 batch R2": r2_by_group(pcs[:, 0], batch),
        "PC2 batch R2": r2_by_group(pcs[:, 1], batch),
        "Silhouette (batch)": silhouette_score(Az, batch),
        "Silhouette (disease)": silhouette_score(Az, disease),
        "SD of Cohen's d across batches": d.std(1).mean(),
        "Sign concordance of d": concord,
    }


metrics = pd.DataFrame({"Before": evaluate(X),
                        "Full ComBat": evaluate(X_full),
                        "Mean-only ComBat": evaluate(X_mo),
                        "Mean-only, no covariate": evaluate(X_nocov)})
metrics.round(4).to_csv(TAB / "batch_correction_metrics.csv")
print(metrics.round(3))

cd = {"before": cohens_d(X), "meanonly": cohens_d(X_mo), "full": cohens_d(X_full)}
for k, v in cd.items():
    v.round(3).to_csv(TAB / f"cohens_d_{k}.csv")
print("Largest |d| under full ComBat:", cd["full"].abs().stack().idxmax(),
      round(cd["full"].abs().stack().max(), 2))

# corrected matrix, for descriptive use only
meta = df[["GSE_ID", "GSM_ID", "GPL", "batch", "subject", "Disease_State", "title"]]
pd.concat([meta.reset_index(drop=True), pd.DataFrame(X_mo, columns=genes)], axis=1) \
  .to_csv(RESULTS / "database_combat_meanonly.csv", index=False)
pd.concat([meta.reset_index(drop=True), pd.DataFrame(X, columns=genes)], axis=1) \
  .to_csv(RESULTS / "database_harmonized_log2.csv", index=False)

# PCA figure
def pca_panel(A, ax, color_by, title, cmap):
    pcs = PCA(2, random_state=0).fit_transform(zscore(A))
    for i, lab in enumerate(sorted(pd.unique(color_by))):
        m = color_by == lab
        ax.scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=.6, label=lab, color=cmap(i))
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
tab10 = plt.get_cmap("tab10")
dcol = lambda i: ["#4C9BE8", "#9E9E9E", "#E05C5C"][i]
pca_panel(X, axes[0, 0], batch, "A  Before correction, coloured by batch", tab10)
pca_panel(X_mo, axes[0, 1], batch, "B  Mean-only ComBat, coloured by batch", tab10)
pca_panel(X, axes[1, 0], disease, "C  Before correction, coloured by disease", dcol)
pca_panel(X_mo, axes[1, 1], disease, "D  Mean-only ComBat, coloured by disease", dcol)
axes[0, 1].legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
axes[1, 1].legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
savefig(fig, "fig_pca_combat")
print("done")
