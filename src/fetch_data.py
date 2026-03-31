"""
Download the Student Dropout and Academic Success dataset from UCI repository.

If data files (features.csv, targets.csv) already exist in the target directory,
this script skips downloading to avoid redundant network requests.
"""

import os


def download_dataset(data_dir: str = "data") -> None:
    """Download UCI dataset (id=697) if CSV files do not already exist.

    Args:
        data_dir: Directory to save the CSV files.
    """
    features_path = os.path.join(data_dir, "features.csv")
    targets_path = os.path.join(data_dir, "targets.csv")

    # 检测是否已有数据文件 / Check if data files already exist
    if os.path.isfile(features_path) and os.path.isfile(targets_path):
        print("Data files already exist, skipping download. / 数据文件已存在，跳过下载。")
        return

    os.makedirs(data_dir, exist_ok=True)

    from ucimlrepo import fetch_ucirepo

    print("Downloading dataset from UCI (id=697)... / 正在从 UCI 下载数据集...")
    dataset = fetch_ucirepo(id=697)

    X = dataset.data.features
    y = dataset.data.targets

    X.to_csv(features_path, index=False)
    y.to_csv(targets_path, index=False)

    print(f"Features saved to {features_path}  (shape: {X.shape})")
    print(f"Targets saved to {targets_path}  (shape: {y.shape})")


if __name__ == "__main__":
    download_dataset()
