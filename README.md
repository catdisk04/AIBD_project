# SLE prediction from genotype and gene-expression data

Code and manuscript for *Prediction of Systemic Lupus Erythematosus from Genotype and Gene-Expression Data: Batch Correction, Leakage-Free Evaluation and the Limits of Multi-Study Pooling* (BT3041 course project, IIT Madras; prepared for bioRxiv).

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
  05_figures.py                 combined ROC / precision-recall figure
results/             every table (tables/) and figure (figures/) in the paper
paper/               earlier manuscript draft (LaTeX)
archive/             original notebooks, the methods draft and the earlier ComBat output
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # on macOS XGBoost also needs: brew install libomp
./run_all.sh                      # ~1 minute; regenerates everything in results/
git status results                # nothing listed = identical to the published results
```

All random seeds are fixed, so a re-run gives byte-identical tables and figures.
Each script can also be run on its own from inside `code/` (scripts 02-05 do not depend
on 01; 05 needs the out-of-fold predictions written by 02 and 03).

| Paper item | Script (`code/`) | Output (`results/tables/` unless noted) |
|---|---|---|
| Table 1 (batches, subjects) | `01_batch_correction.py` | `batch_composition.csv` |
| Table 2 (ComBat diagnostics) | `01_batch_correction.py` | `batch_correction_metrics.csv`, `cohens_d_*.csv` |
| Fig. 1 (PCA) | `01_batch_correction.py` | `figures/fig_pca_combat.pdf` |
| Fig. 2 (SNP QC, association), HWE / chi-square values | `03_snp_analysis.py` | `snp_qc_association.csv`, `figures/fig_snp_qc_association.pdf` |
| Table 3 (SNP classification), paired t-tests | `03_snp_analysis.py` | `snp_cv_summary.csv`, `snp_cv_folds.csv`, `snp_model_ttests.csv` |
| Fig. S1 (SHAP) | `03_snp_analysis.py` | `snp_shap.csv`, `figures/fig_snp_shap.pdf` |
| Table 4 (expression classification, P0-P3, batch-only) | `02_gex_classification.py` | `gex_cv_summary.csv`, `gex_cv_folds.csv`, `gex_model_ttest.csv` |
| Fig. 3 (protocol comparison) | `02_gex_classification.py` | `figures/fig_gex_protocols.pdf` |
| Table 5 (within-batch) | `02_gex_classification.py` | `gex_within_batch.csv`, `gex_within_batch_weighted.csv` |
| Table S1 (leave-one-batch-out) | `02_gex_classification.py` | `gex_lobo.csv` |
| Fig. 4 (ROC and precision-recall curves) | `05_figures.py` | `figures/fig_roc_pr.pdf` (from `oof_*.csv`) |
| Fig. 5, Table S2 (differential expression) | `04_differential_expression.py` | `de_batch_adjusted.csv`, `figures/fig_de_batch_adjusted.pdf` |
