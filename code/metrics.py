"""Evaluation metrics and the CV summary helpers shared by the SNP and GEX scripts."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             f1_score, accuracy_score, balanced_accuracy_score, brier_score_loss)

METRICS = ["ROC-AUC", "PR-AUC", "MCC", "F1", "Balanced acc.", "Accuracy", "Brier"]


def score(y, prob, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(prob) >= thr).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, prob),
        "PR-AUC": average_precision_score(y, prob),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Balanced acc.": balanced_accuracy_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
        "Brier": brier_score_loss(y, prob),
        "Prevalence": y.mean(),
    }


def summarise(fold_df, by="model"):
    """mean and SD across all folds of all repeats."""
    g = fold_df.groupby(by, sort=False)[METRICS + ["Prevalence"]]
    return g.mean().add_suffix(" mean").join(g.std().add_suffix(" sd"))


def fmt_table(fold_df, by="model", cols=METRICS[:5]):
    m = fold_df.groupby(by, sort=False)[cols].mean()
    s = fold_df.groupby(by, sort=False)[cols].std()
    out = m.copy().astype(str)
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(m[c], s[c])]
    return out


def corrected_ttest(a, b, n_train, n_test):
    """
    Nadeau & Bengio (2003) corrected resampled t-test for two models scored on
    the same repeated-CV folds. The plain paired t-test is too liberal because
    the training sets overlap between folds.
    """
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1) * (1.0 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return d.mean(), t, p
