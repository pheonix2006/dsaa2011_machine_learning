# DSAA2011 (L01) Course Project — HKUST(GZ) 2026 Spring

**Student Dropout and Academic Success Dataset** (UCI id=697)

A machine learning pipeline for predicting student outcomes (Dropout / Enrolled / Graduate) using demographic, socioeconomic, and academic features from a Portuguese higher education institution.

## Project Structure

```
2011machine_learning/
├── src/                             # Reusable utility modules
│   ├── fetch_data.py                #   UCI dataset download
│   ├── preprocessing.py             #   Missing value handling, encoding, scaling
│   ├── feature_engineering.py       #   Domain features, target encoding, feature selection
│   ├── visualization.py             #   EDA & t-SNE plotting
│   ├── clustering.py                #   K-Means, hierarchical, DBSCAN, GMM
│   ├── prediction.py                #   Model registry (DT, LR, SVM)
│   └── evaluation.py                #   Metrics, ROC, CV, learning/validation curves
├── notebooks/                       # Jupyter Notebooks (run in order)
│   ├── 01_data_preprocessing.ipynb  #   Data cleaning + feature engineering
│   ├── 02_tsne_visualization.ipynb  #   t-SNE dimensionality reduction
│   ├── 03_clustering_analysis.ipynb #   Clustering (K-Means, hierarchical, DBSCAN, GMM)
│   ├── 04_prediction.ipynb          #   Baseline classifiers (DT, LR, SVM)
│   ├── 05_model_evaluation.ipynb    #   Model evaluation, tuning, ROC, learning curves
│   └── 06_open_ended_exploration.ipynb  # Ensemble models, ablation, bootstrap CI, error analysis
├── data/                            # Generated data files (not version-controlled)
│   ├── features.csv / targets.csv   #   Raw features and labels
│   ├── features_processed.csv       #   Preprocessed features (baseline)
│   ├── features_engineered_*.csv    #   Engineered features (train/val/test splits)
│   ├── targets_engineered_*.csv     #   Corresponding labels (train/val/test splits)
│   ├── *.csv / *.json               #   Experiment results and metrics
│   └── plots/                       #   All generated figures
├── report/                          # LaTeX report (NeurIPS 2025 style)
│   ├── report.tex                   #   Main report source
│   └── build.sh / build.bat         #   Build scripts
├── tests/                           # Unit tests for src modules
├── pyproject.toml                   # Python dependencies (uv)
└── requirements.txt                 # pip-compatible dependencies
```

## Environment Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone <repo-url>
cd 2011machine_learning
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Data Preparation

Data files are not version-controlled (`data/` is in `.gitignore`). Run notebooks in order to generate all data:

1. **01**: Downloads from UCI, applies preprocessing (missing values, encoding, scaling), performs feature engineering (25 domain features), selects top-40 features, and splits into train/val/test (70/15/15)
2. **02–06**: Each notebook loads the engineered data and produces analysis outputs

### Data Splits

| Split | Samples | Usage |
|-------|---------|-------|
| Train | 3,096   | Model training |
| Val   | 664     | Validation / early stopping |
| Test  | 664     | Final held-out evaluation |

All splits use 40 engineered features selected by mutual information.

## Notebooks

| Notebook | Task | Description |
|----------|------|-------------|
| `01_data_preprocessing` | Task 1 | EDA, preprocessing, feature engineering, train/val/test split |
| `02_tsne_visualization` | Task 2 | t-SNE 2D/3D visualization with perplexity comparison |
| `03_clustering_analysis` | Task 3 | K-Means, hierarchical, DBSCAN, GMM clustering |
| `04_prediction` | Task 4 | Baseline classifiers: Decision Tree, Logistic Regression, SVM (RBF) |
| `05_model_evaluation` | Task 5 | ROC-AUC, cross-validation, validation/learning curves, GridSearchCV tuning |
| `06_open_ended_exploration` | Task 6 | Random Forest, Gradient Boosting, feature ablation, bootstrap CI, error analysis |

### Notebook 06 Details

The open-ended exploration notebook covers:
- Ensemble models (Random Forest, Gradient Boosting) with GridSearchCV tuning
- Feature importance analysis (top-10 RF importance)
- Feature selection impact (top-K feature sweep)
- Polynomial interaction features (40 → 85 dim, multiple SelectKBest k values)
- Class weight effect comparison (`balanced` vs. no weighting)
- **Feature group ablation**: academic (20) vs. socioeconomic (20) vs. all (40)
- **Bootstrap confidence intervals**: 10,000 resamples, Tuned RF vs. LR significance test
- **Per-class metrics**: Tuned RF vs. LR per-class precision/recall/F1
- **Enrolled-class error analysis**: DT/LR/SVM confusion matrix decomposition

## Key Results

| Model | Test F1 (macro) |
|-------|----------------|
| Decision Tree | 0.6769 |
| Logistic Regression | 0.7079 |
| SVM (RBF) | 0.6449 |
| Random Forest (tuned) | 0.7096 |
| Gradient Boosting (tuned) | 0.7027 |

- RF and LR achieve comparable performance; bootstrap analysis confirms no statistically significant difference (p = 0.30)
- Academic features provide the primary signal (ablation F1 = 0.671); socioeconomic features add complementary information (combined F1 = 0.717)
- `class_weight='balanced'` is critical for Enrolled class prediction (SVM: 0.00 → 0.47 F1)

## Running Tests

```bash
uv run pytest tests/ -v
```

## Building the Report

```bash
cd report
# Linux/macOS
./build.sh
# Windows
build.bat
```

## License

Course project for educational purposes.
