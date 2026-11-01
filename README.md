# BT3041 Sepsis Project

**Data-driven immune subtypes of sepsis from whole-blood gene expression, and their link to 28-day mortality**
(GEO GSE65682, MARS cohort: 479 sepsis patients, 281 other ICU, 42 healthy controls)

## What is here

| Path | What it is |
|---|---|
| `notebooks/BT3041_Sepsis_Full_Analysis.ipynb` | **Main notebook.** The whole instruction manual, Steps 1–11, run in Google Colab with all outputs, plus the Checkpoint 2 results table |
| `notebooks/01_load_and_qc.ipynb` | Stand-alone loading, cleaning and quality control (Steps 1–2) |
| `notebooks/02_explore_patterns.ipynb` | First pattern analysis (cohort, mortality tests, DE, PCA, Kaplan–Meier) |
| `notebooks/00_rebuild_data.ipynb` | Helper that joins compressed upload pieces back into the CSVs |
| `scripts/01_clean.py`, `scripts/02_patterns.py` | The cleaning and pattern analysis as plain Python scripts |
| `results/` | Tables: cohort Table 1, mortality associations, differential expression, logistic models |
| `figures/` | QC, cohort/mortality, PCA, genes-vs-death and Kaplan–Meier figures |

## Data (not in this repo)

Download from NCBI GEO, accession **GSE65682** (series matrix) and platform **GPL13667**. The notebooks start from two gene-level tables built from them:
`Genes.csv` (genes × 802 samples, log2 RMA) and `Patients.csv` (sample annotations).

## How to run

1. Open `notebooks/BT3041_Sepsis_Full_Analysis.ipynb` in Google Colab.
2. Upload `Genes.csv` and `Patients.csv` in the **Files** panel (`/content`).
3. Runtime → Run all, and allow Google Drive access. Outputs are saved to `MyDrive/BT3041_Sepsis/` (`data/processed/`, `results/`, `figures/`).
Takes about 5–8 minutes on a free Colab runtime.

## Main results (from the Colab run)

| RQ | Answer |
|---|---|
| RQ1 structure | Healthy controls separate from critically ill patients; sepsis forms a continuum ordered by endotype (PC1+PC2 = 37% of variance) |
| RQ2 clusters | No sharp clusters (silhouette ≤ 0.19, gap statistic k = 1); k = 4 partition stable (subsample ARI 0.83), not batch-driven |
| RQ3 mortality | Cluster C1 33% vs 18% 28-day mortality (χ² p = 0.043, log-rank p = 0.029, HR 1.66, 95% CI 1.11–2.50) |
| RQ4 genes | 680 genes FDR < 0.05 (died vs survived), 83% lower in non-survivors (lymphocyte/NK genes) |
| RQ5 pathways | Non-survivors: ↑ haem metabolism, E2F, hypoxia; ↓ interferon α/γ, allograft rejection, inflammatory response |
| RQ6 robustness | Concordant with published Mars endotypes (ARI 0.31 at 2,000 genes; 0.62 at 500) |
| RQ7 confounding | C1 OR 1.76 (1.07–2.88) after age and sex; unchanged with diabetes and hospital |
| RQ8 prediction | Genes beat age: CV AUROC 0.68 vs 0.58 (difference 0.10, 95% CI 0.02–0.18); Amsterdam → Utrecht 0.63 vs 0.59; nested CV 0.68 |
| Extra | Two continuous axes predict death at least as well as discrete groupings (AIC 510 vs 517–519) |

## Software

Python 3, pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, umap-learn, lifelines, gseapy (see `requirements.txt`).

## AI use

AI tools were used to help plan the analysis, explain methods, draft code and review results. All code was run and checked by the team. Prompts and chat histories go in `ai_logs/`.
