"""
Step 3: SNP genotype analysis (QC, per-SNP association, encodings, classification).

Changes from the course report:
  * 10 x 5-fold repeated stratified CV instead of one 5-fold split
  * PR-AUC, MCC, balanced accuracy and Brier score added to ROC-AUC / F1 / accuracy
  * imputation and scaling are refit inside every training fold (as before)
  * the inheritance model is also chosen inside the training fold (nested CV), so
    the reported "best encoding" estimate is not picked on the test folds
  * paired model comparisons use the Nadeau-Bengio corrected resampled t-test
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

from common import DATA, TAB, savefig, SEED
from metrics import score, fmt_table, corrected_ttest, METRICS

N_REPEATS, N_FOLDS = 10, 5
OUTCOME = "Disease out come for only SLE related smp"
COVARS = ["AGE", "sex"]

raw = pd.read_excel(DATA / "Dataset.xls", engine="xlrd")
snps = [c for c in raw.columns if c not in COVARS + [OUTCOME]]
df = raw[raw[OUTCOME].isin([0, 1])].reset_index(drop=True)
y = df[OUTCOME].astype(int).values
print(f"{len(df)} subjects ({y.sum()} SLE, {(1 - y).sum()} controls), {len(snps)} SNPs")
assert not df.duplicated().any()

# ------------------------------------------------------------------ QC
miss = df[snps].isna().mean() * 100


def hwe_exact(het, hom_r, hom_c):
    """Wigginton et al. (2005) exact HWE test."""
    rare = 2 * hom_r + het
    N = hom_r + het + hom_c
    if N == 0 or rare == 0 or rare == 2 * N:
        return 1.0
    probs = [0.0] * (rare + 1)
    mid = int(rare * (2 * N - rare) / (2 * N))
    if (rare % 2) != (mid % 2):
        mid += 1
    probs[mid] = 1.0
    s = 1.0
    hr, hc, hh = (rare - mid) // 2, N - mid - (rare - mid) // 2, mid
    thr, thc, thh = hr, hc, hh
    while thh >= 2:
        probs[thh - 2] = probs[thh] * thh * (thh - 1) / (4 * (thr + 1) * (thc + 1))
        s += probs[thh - 2]; thr += 1; thc += 1; thh -= 2
    thr, thc, thh = hr, hc, hh
    while thh <= rare - 2:
        probs[thh + 2] = probs[thh] * 4 * thr * thc / ((thh + 2) * (thh + 1))
        s += probs[thh + 2]; thr -= 1; thc -= 1; thh += 2
    probs = [p / s for p in probs]
    return min(1.0, sum(p for p in probs if p <= probs[het] + 1e-12))


ctrl = df[y == 0]
hwe = {}
for g in snps:
    c = ctrl[g].dropna()
    c0, c1, c2 = int((c == 0).sum()), int((c == 1).sum()), int((c == 2).sum())
    hwe[g] = hwe_exact(c1, min(c0, c2), max(c0, c2))

# ------------------------------------------------------------------ association
assoc = []
for g in snps:
    ct = pd.crosstab(df[g].dropna(), y[df[g].notna()])
    chi2, p, _, _ = stats.chi2_contingency(ct) if ct.shape[0] > 1 else (np.nan, np.nan, 0, 0)
    assoc.append({"SNP": g, "missing %": miss[g], "HWE p (controls)": hwe[g], "chi2": chi2, "p": p})
assoc = pd.DataFrame(assoc)
ok = assoc.p.notna()
assoc.loc[ok, "q (BH)"] = multipletests(assoc.loc[ok, "p"], method="fdr_bh")[1]
assoc = assoc.sort_values("p")
assoc.to_csv(TAB / "snp_qc_association.csv", index=False, float_format="%.6g")
print(assoc.head(8).round(4))
print(f"Mean missingness {miss.mean():.2f}%, max {miss.max():.2f}% ({miss.idxmax()})")

# ------------------------------------------------------------------ encodings
def enc(kind):
    S = df[snps]
    if kind == "dominant":
        S = S.where(S.isna(), (S >= 1).astype(float))
    elif kind == "recessive":
        S = S.where(S.isna(), (S == 2).astype(float))
    return pd.concat([S, df[COVARS]], axis=1)


X = {"Age + Sex": df[COVARS], "Additive": enc("additive"),
     "Dominant": enc("dominant"), "Recessive": enc("recessive")}


def pipe(kind):
    if kind == "LR":
        return Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("sc", StandardScaler()),
                         ("m", LogisticRegression(C=1, max_iter=5000, random_state=SEED))])
    return Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("m", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                         subsample=0.8, colsample_bytree=0.8,
                                         eval_metric="logloss", random_state=SEED, n_jobs=1))])


configs = [("Age + Sex (LR)", "Age + Sex", "LR"),
           ("Additive + Age + Sex (LR)", "Additive", "LR"),
           ("Dominant + Age + Sex (LR)", "Dominant", "LR"),
           ("Recessive + Age + Sex (LR)", "Recessive", "LR"),
           ("Additive + Age + Sex (XGBoost)", "Additive", "XGB"),
           ("Encoding chosen in inner CV (LR)", None, "LR")]

rows, oof = [], {}
for rep in range(N_REPEATS):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
    folds = list(cv.split(df, y))
    for name, key, model in configs:
        prob = np.zeros(len(y))
        for k, (tr, te) in enumerate(folds):
            if key is None:
                # nested: pick the encoding by inner 5-fold ROC-AUC on the training fold only
                inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
                best = max(["Additive", "Dominant", "Recessive"],
                           key=lambda e: cross_val_score(pipe("LR"), X[e].iloc[tr], y[tr],
                                                         cv=inner, scoring="roc_auc").mean())
                Xk = X[best]
            else:
                Xk = X[key]
            m = pipe(model).fit(Xk.iloc[tr], y[tr])
            p = m.predict_proba(Xk.iloc[te])[:, 1]
            prob[te] = p
            row = {"model": name, "repeat": rep, "fold": k, "n_train": len(tr),
                   "n_test": len(te), **score(y[te], p)}
            if key is None:
                row["chosen"] = best
            rows.append(row)
        if rep == 0:
            oof[name] = prob
    print("repeat", rep, "done")

folds_df = pd.DataFrame(rows)
folds_df.to_csv(TAB / "snp_cv_folds.csv", index=False)
summary = fmt_table(folds_df, cols=METRICS[:6])
summary.to_csv(TAB / "snp_cv_summary.csv")
print(summary)
print("Encoding chosen in inner CV:",
      folds_df.loc[folds_df.chosen.notna(), "chosen"].value_counts().to_dict())
pd.DataFrame({"y": y, **oof}).to_csv(TAB / "oof_snp.csv", index=False)


def paired(a, b):
    A = folds_df[folds_df.model == a].sort_values(["repeat", "fold"])
    B = folds_df[folds_df.model == b].sort_values(["repeat", "fold"])
    out = []
    for m in ["ROC-AUC", "PR-AUC", "MCC"]:
        d, t, p = corrected_ttest(A[m].values, B[m].values, A.n_train.mean(), A.n_test.mean())
        out.append({"comparison": f"{a} vs {b}", "metric": m, "diff": d, "t": t, "p": p})
    return out


tt = pd.DataFrame(paired("Dominant + Age + Sex (LR)", "Additive + Age + Sex (LR)")
                  + paired("Additive + Age + Sex (XGBoost)", "Additive + Age + Sex (LR)")
                  + paired("Dominant + Age + Sex (LR)", "Age + Sex (LR)"))
tt.round(4).to_csv(TAB / "snp_model_ttests.csv", index=False)
print(tt.round(4))

# ------------------------------------------------------------------ SHAP (descriptive)
imp = SimpleImputer(strategy="most_frequent")
Xa = pd.DataFrame(imp.fit_transform(X["Additive"]), columns=X["Additive"].columns)
xgb = pipe("XGB").named_steps["m"].fit(Xa, y)
sv = shap.TreeExplainer(xgb).shap_values(Xa)
shap_mean = pd.Series(np.abs(sv).mean(0), index=Xa.columns).sort_values(ascending=False)
shap_mean.round(4).to_csv(TAB / "snp_shap.csv", header=["mean |SHAP|"])
print(shap_mean.head(6).round(3))

# ------------------------------------------------------------------ Figure: QC & association
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
h = assoc.assign(lp=-np.log10(assoc["HWE p (controls)"].clip(1e-20))).sort_values("lp", ascending=False).head(10)
col = ["#d73027" if p < 1e-3 else "#fc8d59" if p < .05 else "#91bfdb" for p in h["HWE p (controls)"]]
axes[0].barh(h.SNP[::-1], h.lp[::-1], color=col[::-1], height=.65)
axes[0].axvline(3, color="#d73027", ls="--", lw=1, label="p = 0.001")
axes[0].set_xlabel(r"$-\log_{10}$(HWE p-value)")
axes[0].set_title("A  HWE deviations (controls)", loc="left", fontweight="bold")
axes[0].legend(fontsize=8)
a = assoc[assoc.p < 0.065].head(12).assign(lp=lambda d: -np.log10(d.p))
col = ["#2166ac" if q < .05 else "#74add1" for q in a["q (BH)"]]
axes[1].barh(a.SNP[::-1], a.lp[::-1], color=col[::-1], height=.65)
axes[1].axvline(-np.log10(.05), color="#4d4d4d", ls="--", lw=1, label="p = 0.05")
axes[1].set_xlabel(r"$-\log_{10}$($\chi^2$ p-value)")
axes[1].set_title(r"B  Per-SNP association ($\chi^2$, BH-FDR)", loc="left", fontweight="bold")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
savefig(fig, "fig_snp_qc_association")

# SHAP figure
fig, ax = plt.subplots(figsize=(5.5, 4))
s = shap_mean.head(12)[::-1]
ax.barh(s.index, s.values, color="#4C9BE8", height=.65)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("XGBoost (additive + age + sex)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
savefig(fig, "fig_snp_shap")
print("done")
