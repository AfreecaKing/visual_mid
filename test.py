import numpy as np
import pandas as pd
import config
import function

# --- 正規化函數 ---
def minmax_normalize(section):
    """對一個區段做 Min-Max 正規化"""
    min_val = section.min(axis=0)
    max_val = section.max(axis=0)
    denom = np.where(max_val - min_val == 0, 1, max_val - min_val)
    return (section - min_val) / denom

def zscore_normalize(section):
    """對一個區段做 Z-score 標準化"""
    mean = section.mean(axis=0)
    std = section.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (section - mean) / std

# --- 欄位區段正規化 ---
def normalize_features(input_file, output_file, method="minmax"):
    # 定義欄位區段
    feature_sections = {
        "ColorStructure": (0, 31),
        "ColorLayout": (32, 43),
        "RegionShape": (44, 79),
        "HomogeneousTexture": (80, 141),
        "EdgeHistogram": (142, 221)
    }

    # 讀入資料
    all_rows = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_name = parts[0]
            img_path = parts[1]
            features = np.array([float(x) for x in parts[2:]])
            all_rows.append([class_name, img_path, features])

    features_matrix = np.array([row[2] for row in all_rows])

    # 正規化
    features_norm = features_matrix.copy()
    for section_name, (start, end) in feature_sections.items():
        print(f"正在對區段 '{section_name}' (欄位 {start}-{end}) 做 {method} 正規化...")
        section = features_matrix[:, start:end+1]

        if method == "minmax":
            section_norm = minmax_normalize(section)
        elif method == "zscore":
            section_norm = zscore_normalize(section)
        else:
            raise ValueError(f"不支援的正規化方法: {method}")

        features_norm[:, start:end+1] = section_norm

    # 寫出結果
    with open(output_file, 'w') as f:
        for i, row in enumerate(all_rows):
            class_name, img_path = row[0], row[1]
            feat_line = " ".join([f"{x:.6f}" for x in features_norm[i]])
            f.write(f"{class_name} {img_path} {feat_line}\n")

    print(f"{method} 正規化完成，輸出檔案: {output_file}\n")

# --- 範例使用 ---
#normalize_features(config.unformal, "z_fnormal.txt", method="zscore")
#normalize_features(config.unformal, "minmax_formal.txt", method="minmax")
rows = []
with open("minmax_formal.txt", 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        feature, name = parts[0], parts[1]
        nums = list(map(float, parts[2:]))  # 轉浮點數
        rows.append([feature, name] + nums)
formal_df = pd.DataFrame(rows)  # 存成dataframe
formal_df.rename(columns={0: "feature", 1: "name"}, inplace=True)

with open("Cosine_Similarity.txt", 'w', encoding='utf-8') as f:
    df = function.Percision(function.calc_cosine(formal_df))
    for i in range(df.shape[0]):
        f.write(f"name:{df['name'][i]}\n")
        for name in df['neighbors_names'][i]:
            f.write(f"{name}\n")
        f.write(f"{df['percision'][i]}\n")
        f.write("----------------------------\n")
    f.write(f"最終準確率為:{df['percision'].mean()}\n")

with open("Euclidean_Distance.txt", 'w', encoding='utf-8') as f:
    df = function.Percision(function.calc_ed(formal_df))
    for i in range(df.shape[0]):
        f.write(f"name:{df['name'][i]}\n")
        for name in df['neighbors_names'][i]:
            f.write(f"{name}\n")
        f.write(f"{df['percision'][i]}\n")
        f.write("----------------------------\n")
    f.write(f"最終準確率為:{df['percision'].mean()}\n")

with open("PCC.txt", 'w', encoding='utf-8') as f:
    df = function.Percision(function.calc_pcc(formal_df))
    for i in range(df.shape[0]):
        f.write(f"name:{df['name'][i]}\n")
        for name in df['neighbors_names'][i]:
            f.write(f"{name}\n")
        f.write(f"{df['percision'][i]}\n")
        f.write("----------------------------\n")
    f.write(f"最終準確率為:{df['percision'].mean()}\n")