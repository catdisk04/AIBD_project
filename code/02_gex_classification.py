"""
Step 2: SLE vs Healthy Control classification from the 11 ComBat-compatible genes,
evaluated without data leakage, and compared against the protocols that leak.

Protocols (all use 10 x 5-fold CV, the same models and the same metrics):

  P0  Original report : the globally z-scored 13-gene matrix
                        (Datasets/gene_expression_normalized.csv), curated labels,
                        folds split by sample.
  P1  Global ComBat   : mean-only ComBat with the disease covariate fitted once on
                        all samples, then folds split by sample.
  P2  Global ComBat,
      grouped folds   : as P1, but every person's samples stay in the same fold.
  P3  Leakage-free    : grouped folds, and gene filtering, ComBat, scaling and the
      (primary)         model are all fitted on the training fold only. The test fold
                        is corrected with the training-fold batch shifts, which need
                        no labels.

Also: leave-one-batch-out (LOBO) for every batch that contains both classes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve
from xgboost import XGBClassifier

from common import (load_expression, harmonize_scale, combat_ok_genes, disease_covariates,
                    savefig, TAB, RESULTS, DATA, ALL_GENES, SEED)
from combat import ComBat
from metrics import score, fmt_table, corrected_ttest, METRICS

N_REPEATS, N_FOLDS = 10, 5


def make_model(kind):
    if kind == "L1-LR":
        return LogisticRegression(C=1, penalty="l1", solver="liblinear", max_iter=5000,
                                  random_state=SEED)
    if kind == "Batch only":
        return LogisticRegression(C=1, max_iter=5000, random_state=SEED)
    if kind == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                             n_jobs=1)
    raise ValueError(kind)


def fit_predict(kind, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = make_model(kind).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1], m


# ---------------------------------------------------------------- data
df = harmonize_scale(load_expression())
lab = df["Disease_State"].isin(["SLE", "Healthy Control"]).values
y_all = (df["Disease_State"] == "SLE").astype(int).values
batch = df["batch"].values
subject = df["subject"].values
covar = disease_covariates(df["Disease_State"])
genes = combat_ok_genes(df)

idx_lab = np.where(lab)[0]                 # the samples we classify
idx_other = np.where(~lab)[0]              # Other/Unknown: only ever used to fit ComBat
y = y_all[idx_lab]
strata = np.char.add(np.char.add(batch[idx_lab].astype(str), "|"), y.astype(str))
print(f"Classification set: {len(idx_lab)} samples from {len(set(subject[idx_lab]))} subjects "
      f"(SLE={y.sum()}, HC={(1 - y).sum()})")

# global (leaky) ComBat for P1/P2
X_global = ComBat(mean_only=True).fit_transform(df[genes].values, batch, covar)[idx_lab]

# original report matrix for P0
orig = pd.read_csv(DATA / "gene_expression_normalized.csv")
orig = orig[orig["Disease_State"].isin(["SLE", "Healthy Control"])].reset_index(drop=True)
X_orig = orig[ALL_GENES].values
y_orig = (orig["Disease_State"] == "SLE").astype(int).values


def splits(n, yy, grouped, rep):
    if grouped:
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        return cv.split(np.zeros(n), strata, subject[idx_lab])
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
    return cv.split(np.zeros(n), yy)


def fold_leakage_free(tr, te):
    """P3: everything learned from the training fold (+ unlabelled Other samples)."""
    fit_idx = np.concatenate([idx_lab[tr], idx_other])
    sub = df.iloc[fit_idx]
    g = combat_ok_genes(sub, genes)
    cb = ComBat(mean_only=True)
    Xfit = cb.fit_transform(sub[g].values, batch[fit_idx], covar[fit_idx])
    Xtr = Xfit[: len(tr)]
    Xte = cb.transform(df.iloc[idx_lab[te]][g].values, batch[idx_lab[te]])
    return Xtr, Xte


BATCHES = sorted(pd.unique(batch))


def onehot(b):
    return (np.asarray(b)[:, None] == np.array(BATCHES)[None, :]).astype(float)


rows, oof, oof_p3 = [], {}, []
for proto in ["P0", "P1", "P2", "P3"]:
    kinds = ["L1-LR", "XGBoost"] + (["Batch only"] if proto == "P3" else [])
    for kind in kinds:
        for rep in range(N_REPEATS):
            yy = y_orig if proto == "P0" else y
            n = len(yy)
            prob = np.zeros(n)
            for k, (tr, te) in enumerate(splits(n, yy, proto in ("P2", "P3"), rep)):
                if proto in ("P2", "P3"):
                    assert not set(subject[idx_lab[tr]]) & set(subject[idx_lab[te]])
                if kind == "Batch only":
                    Xtr, Xte = onehot(batch[idx_lab[tr]]), onehot(batch[idx_lab[te]])
                elif proto == "P0":
                    Xtr, Xte = X_orig[tr], X_orig[te]
                elif proto in ("P1", "P2"):
                    Xtr, Xte = X_global[tr], X_global[te]
                else:
                    Xtr, Xte = fold_leakage_free(tr, te)
                p, _ = fit_predict(kind, Xtr, yy[tr], Xte)
                prob[te] = p
                rows.append({"protocol": proto, "model": kind, "repeat": rep, "fold": k,
                             "n_train": len(tr), "n_test": len(te), **score(yy[te], p)})
            if rep == 0:
                oof[(proto, kind)] = (yy.copy(), prob.copy())
            if proto == "P3":
                oof_p3.append(pd.DataFrame({"model": kind, "repeat": rep, "y": yy, "prob": prob,
                                            "batch": batch[idx_lab]}))
        print(proto, kind, "done")

folds = pd.DataFrame(rows)
folds.to_csv(TAB / "gex_cv_folds.csv", index=False)
folds["key"] = folds["protocol"] + " / " + folds["model"]
summary = fmt_table(folds, by="key", cols=METRICS[:6])
summary.to_csv(TAB / "gex_cv_summary.csv")
print(summary)

# pooled out-of-fold predictions of the primary protocol, for the figures
for (proto, kind), (yy, pp) in oof.items():
    pd.DataFrame({"y": yy, "prob": pp}).to_csv(TAB / f"oof_gex_{proto}_{kind}.csv", index=False)

# within-batch discrimination of the leakage-free models: does the model separate
# SLE from HC inside a study, or only tell the studies apart?
oof_p3 = pd.concat(oof_p3)
wb = []
for (kind, rep, b), g in oof_p3.groupby(["model", "repeat", "batch"]):
    if g.y.nunique() == 2:
        wb.append({"model": kind, "repeat": rep, "batch": b, "n": len(g), "n SLE": g.y.sum(),
                   **score(g.y.values, g.prob.values)})
wb = pd.DataFrame(wb)
wb_sum = wb.groupby(["model", "batch"])[["n", "n SLE", "ROC-AUC", "PR-AUC", "MCC", "Prevalence"]].mean()
wb_sum.round(4).to_csv(TAB / "gex_within_batch.csv")
print(wb_sum.round(3))
# sample-size-weighted mean within-batch AUC, per repeat
wavg = wb.groupby(["model", "repeat"]).apply(
    lambda g: pd.Series({m: np.average(g[m], weights=g["n"]) for m in ["ROC-AUC", "PR-AUC", "MCC"]}))
wavg = wavg.groupby("model").agg(["mean", "std"])
wavg.round(4).to_csv(TAB / "gex_within_batch_weighted.csv")
print(wavg.round(3))

# how much each leak inflates performance (L1-LR, paired on repeat/fold)
infl = []
p3 = folds[(folds.protocol == "P3") & (folds.model == "L1-LR")].set_index(["repeat", "fold"])
for proto in ["P1", "P2"]:
    other = folds[(folds.protocol == proto) & (folds.model == "L1-LR")].set_index(["repeat", "fold"])
    for m in ["ROC-AUC", "PR-AUC", "MCC"]:
        infl.append({"comparison": f"{proto} - P3", "metric": m,
                     "mean difference": other[m].mean() - p3[m].mean()})
pd.DataFrame(infl).round(4).to_csv(TAB / "gex_leakage_inflation.csv", index=False)

# LR vs XGBoost in the leakage-free protocol (same folds -> corrected t-test)
a = folds[(folds.protocol == "P3") & (folds.model == "XGBoost")].sort_values(["repeat", "fold"])
b = folds[(folds.protocol == "P3") & (folds.model == "L1-LR")].sort_values(["repeat", "fold"])
tt = []
for m in ["ROC-AUC", "PR-AUC", "MCC"]:
    d, t, pv = corrected_ttest(a[m].values, b[m].values, a.n_train.mean(), a.n_test.mean())
    tt.append({"comparison": "XGBoost - L1-LR (P3)", "metric": m, "diff": d, "t": t, "p": pv})
pd.DataFrame(tt).round(4).to_csv(TAB / "gex_model_ttest.csv", index=False)
print(pd.DataFrame(tt).round(4))

# ---------------------------------------------------------------- final model coefficients
# (L1-LR fitted on all labelled data after mean-only ComBat; descriptive only)
fit_idx = np.concatenate([idx_lab, idx_other])
Xall = ComBat(mean_only=True).fit_transform(df[genes].values[fit_idx], batch[fit_idx], covar[fit_idx])
_, mfull = fit_predict("L1-LR", Xall[: len(idx_lab)], y, Xall[: len(idx_lab)])
pd.Series(mfull.coef_[0], index=genes, name="L1-LR coefficient (standardised)") \
  .round(4).to_csv(TAB / "gex_l1lr_coefficients.csv")

# ---------------------------------------------------------------- leave-one-batch-out
lobo = []
both = [b for b in pd.unique(batch[idx_lab])
        if len(set(y[batch[idx_lab] == b])) == 2]
for b in both:
    te = np.where(batch[idx_lab] == b)[0]
    tr = np.where(batch[idx_lab] != b)[0]
    fit_idx = np.concatenate([idx_lab[tr], idx_other[batch[idx_other] != b]])
    sub = df.iloc[fit_idx]
    cb = ComBat(mean_only=True)
    Xfit = cb.fit_transform(sub[genes].values, batch[fit_idx], covar[fit_idx])
    Xtr = Xfit[: len(tr)]
    # unseen batch: label-free shift of its gene means onto the training mean
    Xte_raw = df.iloc[idx_lab[te]][genes].values
    Xte = Xte_raw - Xte_raw.mean(0) + Xtr.mean(0)
    for kind in ["L1-LR", "XGBoost"]:
        p, _ = fit_predict(kind, Xtr, y[tr], Xte)
        lobo.append({"held-out batch": b, "model": kind, "n": len(te),
                     "n SLE": int(y[te].sum()), **score(y[te], p)})
lobo = pd.DataFrame(lobo)
lobo.round(4).to_csv(TAB / "gex_lobo.csv", index=False)
print(lobo[["held-out batch", "model", "n", "n SLE", "ROC-AUC", "PR-AUC", "MCC"]].round(3))

# ---------------------------------------------------------------- figure: protocols
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
lab_map = {"P0": "P0\noriginal", "P1": "P1\nglobal ComBat", "P2": "P2\n+ subject CV",
           "P3": "P3\nleakage-free"}
bonly = folds[(folds.protocol == "P3") & (folds.model == "Batch only")]
l1 = folds[folds.model == "L1-LR"]
for a_, m in zip(ax, ["ROC-AUC", "PR-AUC", "MCC"]):
    data = [l1.loc[l1.protocol == p, m].values for p in lab_map]
    bp = a_.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#bdbdbd", "#fdae61", "#abd9e9", "#2c7bb6"]):
        patch.set_facecolor(c)
    a_.axhline(bonly[m].mean(), color="#555555", ls=":", lw=1.2, label="batch label only")
    a_.set_xticks(range(1, 5), [lab_map[p] for p in lab_map], fontsize=8)
    a_.set_title(m, fontsize=10)
    a_.spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle("GEX L1-LR under each evaluation protocol (10 x 5-fold CV)", fontsize=10)
fig.tight_layout()
savefig(fig, "fig_gex_protocols")
print("done")
