# DSAA2011 (L01) Course Project — HKUST(GZ) 2026 Spring

**Student Dropout and Academic Success Dataset** (UCI id=697)

## Project Structure

```
dsaa2011_machine_learning/
├── src/                         # 可复用工具函数
│   ├── fetch_data.py            #   数据下载
│   ├── preprocessing.py          #   数据预处理
│   ├── feature_engineering.py    #   特征工程
│   ├── visualization.py          #   数据可视化
│   ├── clustering.py             #   聚类分析
│   ├── prediction.py             #   预测模型
│   └── evaluation.py             #   模型评估
├── notebooks/                   # Jupyter Notebooks（按顺序运行）
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_tsne_visualization.ipynb
│   ├── 03_clustering_analysis.ipynb
│   ├── 04_prediction.ipynb
│   └── 05_model_evaluation.ipynb
├── data/                        # 数据文件（已加入 .gitignore）
│   ├── features.csv              #   原始特征
│   ├── targets.csv              #   原始标签
│   ├── features_processed.csv   #   处理后特征（baseline）
│   ├── targets_processed.csv     #   处理后标签
│   ├── features_engineered.csv   #   特征工程后特征
│   ├── targets_engineered.csv    #   特征工程对应标签
│   └── plots/                   #   所有生成的图表
├── pyproject.toml               # Python 依赖配置
└── project_announce_L01.md     # 项目任务要求
```

## Environment Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone <repo-url>
cd dsaa2011_machine_learning
uv sync
```

## Data Preparation

数据文件未纳入版本控制（`data/` 已加入 `.gitignore`）。首次使用按顺序运行 Notebooks 即可：

1. **数据下载** — 通过 UCI ML Repo 自动下载原始数据
2. **数据预处理** — 缺失值处理、特征编码（one-hot）、标准化
3. **特征工程** — domain features、target encoding、top-60 特征选择
4. **数据保存** — 生成 `features_processed.csv`（baseline）和 `features_engineered.csv`（工程化）

切换数据集：在对应 Notebook 数据加载 cell 中注释/取消注释相应行即可。

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_preprocessing.ipynb` | 数据预处理 + 特征工程 |
| `02_tsne_visualization.ipynb` | t-SNE 降维可视化 |
| `03_clustering_analysis.ipynb` | 聚类分析 |
| `04_prediction.ipynb` | 预测模型训练与测试 |
| `05_model_evaluation.ipynb` | 模型评估与选择 |

## License

Course project for educational purposes.
