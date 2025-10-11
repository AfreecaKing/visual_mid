import numpy as np
import config
import glob
import os

os.makedirs(config.processed_data_path, exist_ok=True)

# 第一階段：收集所有特徵
all_data = []
all_names = []
all_picture_content = []

for filename in glob.glob(config.original_data_path + "/*.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().split()
        for i in range(201):
            if content[i * 482 + 480][-3:] == "jpg":
                name = content[i * 482 + 480]
                features = content[i * 482 + 1:i * 482 + 81] + content[i * 482 + 337:i * 482 + 479]
                all_names.append(name)
                all_picture_content.append(features)

# 轉換為 numpy array
data = np.array(all_picture_content, dtype=float)
# ==================== 1. 全域 Z-score 正規化 ====================
mean_global = np.mean(data, axis=0)  # shape: (222,)
std_global = np.std(data, axis=0)  # shape: (222,)
std_global[std_global == 0] = 1e-8  # 避免除以零
z_score_normalized = (data - mean_global) / std_global

# ==================== 2. 全域 Max 正規化 ====================
max_abs_global = np.max(np.abs(data), axis=0)  # shape: (222,)
max_abs_global[max_abs_global == 0] = 1e-8  # 避免除以零
max_normalized = data / max_abs_global


# ==================== 寫入檔案 ====================

# 寫入未正規化資料
with open(config.unformal, 'w', encoding='utf-8') as f:
    for i, name in enumerate(all_names):
        f.write(name[:-8] + ' ')
        f.write(name + ' ')
        f.write(' '.join(all_picture_content[i]) + '\n')

# 寫入 Z-score 正規化資料
with open(config.z_score, 'w', encoding='utf-8') as f:
    for i, name in enumerate(all_names):
        f.write(name[:-8] + ' ')
        f.write(name + ' ')
        f.write(' '.join(f"{x:.4f}" for x in z_score_normalized[i]) + '\n')

# 寫入 Max 正規化資料
with open(config.max, 'w', encoding='utf-8') as f:
    for i, name in enumerate(all_names):
        f.write(name[:-8] + ' ')
        f.write(name + ' ')
        f.write(' '.join(f"{x:.4f}" for x in max_normalized[i]) + '\n')

