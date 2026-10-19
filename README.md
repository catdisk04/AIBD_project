# SLE prediction from genotype and gene-expression data

Code and data for *Prediction of Systemic Lupus Erythematosus from Genotype and Gene-Expression Data: Batch Correction and Leakage-Free Evaluation* (BT3041 course project, IIT Madras; prepared for bioRxiv).

**Authors:** Vasudha Kannan, R Karthik, Aldis Daniel, Hitesh Kandarpa (Indian Institute of Technology Madras)

## Layout

```
Datasets/            input data
  Dataset.xls          SNP genotypes (425 subjects, 30 SNPs + age, sex)
  database.csv         curated GEO expression, raw values (1,211 samples, 13 genes)
  gene_expression_normalized.csv   labels + the z-scored matrix used by the original report
code/
  common.py            loading, scale harmonisation, subject IDs, gene filter
  combat.py            ComBat with fit/transform (mean-only transform needs no labels)
  metrics.py           ROC-AUC, PR-AUC, MCC, ... and the corrected resampled t-test
  01_batch_correction.py        ComBat + diagnostics (Table 2, Fig. 1)
  02_gex_classification.py      leakage-free CV, leaky-protocol comparison, batch-only
                                baseline, within-batch and leave-one-batch-out
  03_snp_analysis.py            SNP QC, association, encodings, nested CV, SHAP
  04_differential_expression.py batch-adjusted DE with subject-clustered SEs
  05_figures.py                 combined ROC / precision-recall and SHAP figures
  06_sanity_checks.py           label-permutation checks of the whole pipeline
  gexlib.py                     shared expression CV pipeline used by 02 and 06
results/             every table (tables/) and figure (figures/) in the paper
archive/             original notebooks, the methods draft and the earlier ComBat output
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # on macOS XGBoost also needs: brew install libomp
./run_all.sh                      # ~5 minutes; regenerates everything in results/
git status results                # nothing listed = identical to the published results
```

All random seeds are fixed, so a re-run gives byte-identical tables and figures.
Each script can also be run on its own from inside `code/` (02-04 and 06 do not depend on
01; 05 needs the out-of-fold predictions and SHAP values written by 02, 03 and 04).

| Paper item | Script (`code/`) | Output (`results/tables/` unless noted) |
|---|---|---|
| Table 1 (batches, subjects) | `01_batch_correction.py` | `batch_composition.csv` |
| Table 2 (ComBat diagnostics) | `01_batch_correction.py` | `batch_correction_metrics.csv`, `cohens_d_*.csv` |
| Fig. 1 (PCA) | `01_batch_correction.py` | `figures/fig_pca_combat.pdf` |
| Fig. 2 (SNP QC, association), HWE / chi-square values | `03_snp_analysis.py` | `snp_qc_association.csv`, `figures/fig_snp_qc_association.pdf` |
| Table 3 (genotype counts, odds ratios) | `03_snp_analysis.py` | `snp_effects.csv` |
| Table 4 (SNP classification), paired t-tests | `03_snp_analysis.py` | `snp_cv_summary.csv`, `snp_cv_folds.csv`, `snp_model_ttests.csv` |
| Table 5 (pathway models, HWE sensitivity) | `03_snp_analysis.py` | `snp_cv_summary.csv`, `snp_model_ttests.csv` |
| Table 6 (expression classification, P0-P3, batch-only) | `02_gex_classification.py` | `gex_cv_summary.csv`, `gex_cv_folds.csv`, `gex_model_ttest.csv` |
| Fig. 3 (protocol comparison) | `02_gex_classification.py` | `figures/fig_gex_protocols.pdf` |
| Table 7 (within-batch) | `02_gex_classification.py` | `gex_within_batch.csv`, `gex_within_batch_weighted.csv` |
| Table 8 (expression SHAP, L1-LR coefficients, univariate AUCs) | `02_gex_classification.py` | `gex_explanations.csv`, `gex_shap_oof.csv` |
| Fig. 4 (ROC and precision-recall curves) | `05_figures.py` | `figures/fig_roc_pr.pdf` (from `oof_*.csv`) |
| Fig. 5, Table S2 (differential expression) | `04_differential_expression.py` | `de_batch_adjusted.csv`, `figures/fig_de_batch_adjusted.pdf` |
| Fig. 6 (SHAP, both models) | `05_figures.py` | `figures/fig_shap.pdf` (from `snp_shap_oof.csv`, `gex_shap_oof.csv`) |
| Table S1 (leave-one-batch-out) | `02_gex_classification.py` | `gex_lobo.csv` |
| Fig. S1, permutation checks | `06_sanity_checks.py` | `permutation_summary.csv`, `permutation_runs.csv`, `figures/fig_permutation.pdf` |
