"""
Step 4: SLE vs Healthy Control differential expression.

Following Nygaard et al. (2016) we test on the harmonised (log2-like) data with batch
in the model, rather than on the ComBat output, so the test does not inherit the
false confidence that comes from correcting for batch with the disease covariate.
Model per gene:  expression ~ SLE + batch, with standard errors clustered by subject
because several people contribute more than one sample.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from common import load_expression, harmonize_scale, combat_ok_genes, savefig, TAB

df = harmonize_scale(load_expression())
genes = combat_ok_genes(df)
sub = df[df["Disease_State"].isin(["SLE", "Healthy Control"])].copy()
sub["SLE"] = (sub["Disease_State"] == "SLE").astype(float)
B = pd.get_dummies(sub["batch"], drop_first=True).astype(float)
design = sm.add_constant(pd.concat([sub[["SLE"]], B], axis=1))
groups = pd.factorize(sub["subject"])[0]

rows = []
for g in genes:
    fit = sm.OLS(sub[g].astype(float), design).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int().loc["SLE"]
    rows.append({"gene": g, "log2FC (SLE - HC)": fit.params["SLE"], "CI low": ci[0],
                 "CI high": ci[1], "SE (cluster)": fit.bse["SLE"], "p": fit.pvalues["SLE"]})
res = pd.DataFrame(rows).set_index("gene")
res["q (BH)"] = multipletests(res["p"], method="fdr_bh")[1]
res = res.sort_values("p")
res.round(6).to_csv(TAB / "de_batch_adjusted.csv")
print(res.round(4))
print(f"{(res['q (BH)'] < 0.05).sum()} / {len(res)} genes at q < 0.05")

fig, ax = plt.subplots(figsize=(5.5, 4.2))
r = res.sort_values("log2FC (SLE - HC)")
for i, (g, row) in enumerate(r.iterrows()):
    sig = row["q (BH)"] < 0.05
    c = "#d73027" if row["log2FC (SLE - HC)"] > 0 else "#2166ac"
    a = 1.0 if sig else 0.35
    ax.plot([row["CI low"], row["CI high"]], [i, i], color=c, alpha=a, lw=1.6)
    ax.scatter(row["log2FC (SLE - HC)"], i, color=c, alpha=a, s=36, zorder=3)
ax.axvline(0, color="k", lw=.7)
ax.set_yticks(range(len(r)), r.index, fontsize=8)
ax.set_xlabel(r"log$_2$FC, SLE $-$ HC (batch-adjusted, 95% CI)")
ax.legend(handles=[mpatches.Patch(color="#d73027", label="Up in SLE (q < 0.05)"),
                   mpatches.Patch(color="#2166ac", label="Down in SLE (q < 0.05)"),
                   mpatches.Patch(color="#aaa", alpha=.4, label="q ≥ 0.05")], fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
savefig(fig, "fig_de_batch_adjusted")
