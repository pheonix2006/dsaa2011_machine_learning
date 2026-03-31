# DSAA2011 (L01) Course Project — HKUST(GZ) 2026 Spring

**Student Dropout and Academic Success Dataset** (UCI id=697)

## Project Structure

```
2011machine_learning/
├── src/                         # 可复用工具函数
│   ├── fetch_data.py            #   数据下载
│   ├── preprocessing.py         #   数据预处理
│   ├── visualization.py         #   数据可视化
│   ├── clustering.py            #   聚类分析
│   └── models.py                #   预测模型
├── notebooks/                   # Jupyter Notebooks (每个任务一个)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_tsne_visualization.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_prediction.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_open_exploration.ipynb
├── data/                        # 数据文件 (已加入 .gitignore)
│   ├── features.csv             #   原始特征
│   ├── targets.csv              #   原始标签
│   ├── features_processed.csv   #   处理后特征
│   ├── targets_processed.csv    #   处理后标签
│   └── plots/                   #   所有生成的图表
├── pyproject.toml               # Python 依赖配置
└── project_announce_L01.md      # 项目任务要求
```

## Environment Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd 2011machine_learning

# Create virtual environment and install dependencies (using uv)
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -e .
```

## Data Preparation

数据文件未纳入版本控制（`data/` 已加入 `.gitignore`）。首次使用只需按顺序运行 Notebooks 即可：

打开 `notebooks/01_data_preprocessing.ipynb` 并按顺序执行所有单元格，该 Notebook 会自动完成以下步骤：

1. **数据下载** — 通过 UCI ML Repo 自动下载原始数据（若已存在则跳过）
2. **数据加载** — 读取 `data/features.csv` 和 `data/targets.csv`
3. **数据预处理** — 缺失值处理、特征编码（one-hot）、标准化
4. **数据保存** — 生成 `data/features_processed.csv` 和 `data/targets_processed.csv`
5. **可视化** — 生成统计图表至 `data/plots/`

> 完成后即可按编号顺序运行其余 Notebook。

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_preprocessing.ipynb` | 数据加载、缺失值处理、特征编码与标准化 |
| `02_tsne_visualization.ipynb` | t-SNE 降维可视化 |
| `03_clustering.ipynb` | 聚类分析 |
| `04_prediction.ipynb` | 预测模型训练与测试 |
| `05_model_evaluation.ipynb` | 模型评估与选择 |
| `06_open_exploration.ipynb` | 开放式探索 |

## License

Course project for educational purposes.
