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
paper/               main.tex, refs.bib, main.pdf
archive/             original notebooks, the methods draft and the earlier ComBat output
```

## Reproduce

```bash
pip install -r requirements.txt   # on macOS XGBoost also needs: brew install libomp
./run_all.sh                      # ~1 minute; regenerates results/
cd paper && tectonic main.tex     # or pdflatex + bibtex
```
