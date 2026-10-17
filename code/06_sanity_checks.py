"""
Step 6: label-permutation checks of the whole evaluation pipeline.

If a pipeline leaks test information, it can score above chance even when the labels
carry no signal. We therefore re-run the full pipelines with shuffled labels:

  SNP, dominant LR       : labels shuffled across subjects -> expect ROC-AUC ~ 0.5
  GEX, P3 (L1-LR, XGB)   : (a) labels shuffled across all samples (ComBat also gets the
                               shuffled labels) -> expect ROC-AUC ~ 0.5
                           (b) labels shuffled only within each batch, which keeps the
                               batch-label confounding but removes any gene-disease link
                               inside a batch -> pooled ROC-AUC should approach the
                               batch-only model, within-batch ROC-AUC should be ~ 0.5

The permutation p-value is (1 + #null >= observed) / (1 + #permutations).
One 5-fold split (the first repeat of the main analysis) is used per permutation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from common import DATA, TAB, SEED, savefig, disease_covariates
from gexlib import (fit_predict, batch, idx_lab, idx_other, y, splits, fold_leakage_free,
                    covar)

N_PERM = 100
rng = np.random.default_rng(SEED)
rows = []

# ------------------------------------------------------------------ SNP
OUTCOME = "Disease out come for only SLE related smp"
raw = pd.read_excel(DATA / "Dataset.xls", engine="xlrd")
d = raw[raw[OUTCOME].isin([0, 1])].reset_index(drop=True)
ys = d[OUTCOME].astype(int).values
snps = [c for c in d.columns if c not in ["AGE", "sex", OUTCOME]]
S = d[snps]
Xs = pd.concat([S.where(S.isna(), (S >= 1).astype(float)), d[["AGE", "sex"]]], axis=1)
folds_s = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, ys))


def snp_auc(yy):
    prob = np.zeros(len(yy))
    for tr, te in folds_s:
        m = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("sc", StandardScaler()),
                      ("m", LogisticRegression(C=1, max_iter=5000, random_state=SEED))])
        prob[te] = m.fit(Xs.iloc[tr], yy[tr]).predict_proba(Xs.iloc[te])[:, 1]
    return roc_auc_score(yy, prob)


rows.append({"analysis": "SNP dominant LR", "scheme": "observed", "perm": -1,
             "pooled ROC-AUC": snp_auc(ys)})
for i in range(N_PERM):
    rows.append({"analysis": "SNP dominant LR", "scheme": "shuffled", "perm": i,
                 "pooled ROC-AUC": snp_auc(rng.permutation(ys))})
print("SNP done")

# ------------------------------------------------------------------ GEX (protocol P3)
folds_g = list(splits(len(y), y, True, 0))
b_lab = batch[idx_lab]
both = [b for b in pd.unique(b_lab) if len(set(y[b_lab == b])) == 2]
w = np.array([(b_lab == b).sum() for b in both])


def gex_eval(kind, yy):
    # ComBat sees the same (possibly shuffled) labels as the classifier
    labels = np.where(yy == 1, "SLE", "Healthy Control")
    cov = covar.copy()
    cov[idx_lab] = disease_covariates(labels)
    prob = np.zeros(len(yy))
    for tr, te in folds_g:
        Xtr, Xte = fold_leakage_free(tr, te, covar_fit=cov)
        prob[te] = fit_predict(kind, Xtr, yy[tr], Xte)[0]
    within = np.average([roc_auc_score(yy[b_lab == b], prob[b_lab == b]) for b in both],
                        weights=w)
    return roc_auc_score(yy, prob), within


def shuffle_within_batch(yy):
    out = yy.copy()
    for b in pd.unique(b_lab):
        m = np.where(b_lab == b)[0]
        out[m] = rng.permutation(yy[m])
    return out


for kind in ["L1-LR", "XGBoost"]:
    pooled, within = gex_eval(kind, y)
    rows.append({"analysis": f"GEX P3 {kind}", "scheme": "observed", "perm": -1,
                 "pooled ROC-AUC": pooled, "within-batch ROC-AUC": within})
    for scheme, fn in [("shuffled", rng.permutation), ("shuffled within batch", shuffle_within_batch)]:
        for i in range(N_PERM):
            pooled, within = gex_eval(kind, fn(y))
            rows.append({"analysis": f"GEX P3 {kind}", "scheme": scheme, "perm": i,
                         "pooled ROC-AUC": pooled, "within-batch ROC-AUC": within})
    print(kind, "done")

perm = pd.DataFrame(rows)
perm.to_csv(TAB / "permutation_runs.csv", index=False, float_format="%.6g")

summ = []
for (a, sch), g in perm[perm.scheme != "observed"].groupby(["analysis", "scheme"]):
    obs = perm[(perm.analysis == a) & (perm.scheme == "observed")].iloc[0]
    r = {"analysis": a, "null scheme": sch, "n permutations": len(g)}
    for m in ["pooled ROC-AUC", "within-batch ROC-AUC"]:
        if g[m].notna().all() and pd.notna(obs.get(m)):
            r[f"{m} observed"] = obs[m]
            r[f"{m} null mean"] = g[m].mean()
            r[f"{m} null 95th pct"] = g[m].quantile(0.95)
            r[f"{m} p"] = (1 + (g[m] >= obs[m]).sum()) / (1 + len(g))
    summ.append(r)
summ = pd.DataFrame(summ)
summ.round(4).to_csv(TAB / "permutation_summary.csv", index=False)
print(summ.round(3).T)

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
panels = [("SNP dominant LR", "pooled ROC-AUC", "A  SNP, dominant LR"),
          ("GEX P3 L1-LR", "pooled ROC-AUC", "B  Expression L1-LR, pooled"),
          ("GEX P3 L1-LR", "within-batch ROC-AUC", "C  Expression L1-LR, within batch")]
cols = {"shuffled": "#9e9e9e", "shuffled within batch": "#fdae61"}
for a_, (an, m, title) in zip(ax, panels):
    sub = perm[perm.analysis == an]
    for sch, c in cols.items():
        v = sub.loc[sub.scheme == sch, m].dropna()
        if len(v):
            a_.hist(v, bins=20, color=c, alpha=.8, label=f"null: {sch}")
    o = sub.loc[sub.scheme == "observed", m].iloc[0]
    a_.axvline(o, color="#c62828", lw=2, label=f"observed ({o:.3f})")
    a_.set_title(title, loc="left", fontsize=10, fontweight="bold")
    a_.set_xlabel("ROC-AUC")
    a_.spines[["top", "right"]].set_visible(False)
    a_.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "fig_permutation")
print("done")
