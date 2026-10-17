"""
Shared pieces of the expression classification: the data set up for SLE vs control,
the subject-grouped folds, the leakage-free fold (gene filter + ComBat fitted on the
training fold only) and the two classifiers. Used by 02_gex_classification.py and
06_sanity_checks.py so that both run exactly the same pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import load_expression, harmonize_scale, combat_ok_genes, disease_covariates, SEED
from combat import ComBat

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
    return m.predict_proba(sc.transform(Xte))[:, 1], m, sc


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

BATCHES = sorted(pd.unique(batch))


def onehot(b):
    return (np.asarray(b)[:, None] == np.array(BATCHES)[None, :]).astype(float)


def splits(n, yy, grouped, rep):
    if grouped:
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        return cv.split(np.zeros(n), strata, subject[idx_lab])
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
    return cv.split(np.zeros(n), yy)


def fold_leakage_free(tr, te, covar_fit=None):
    """
    P3: everything learned from the training fold (+ unlabelled Other samples).
    covar_fit lets the permutation checks pass permuted labels to ComBat, so the
    whole pipeline, not just the classifier, sees the shuffled outcome.
    """
    cv_ = covar if covar_fit is None else covar_fit
    fit_idx = np.concatenate([idx_lab[tr], idx_other])
    sub = df.iloc[fit_idx]
    g = combat_ok_genes(sub, genes)
    cb = ComBat(mean_only=True)
    Xfit = cb.fit_transform(sub[g].values, batch[fit_idx], cv_[fit_idx])
    Xtr = Xfit[: len(tr)]
    Xte = cb.transform(df.iloc[idx_lab[te]][g].values, batch[idx_lab[te]])
    assert g == genes, "gene filter dropped a gene inside a fold"
    return Xtr, Xte
