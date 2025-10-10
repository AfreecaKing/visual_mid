import config
import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np

# 把processed features內的txt資料讀出來
# 讀取unformal
rows = []
with open(config.unformal, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        feature, name = parts[0], parts[1]
        nums = list(map(float, parts[2:]))  # 轉浮點數
        rows.append([feature, name] + nums)
unformal_df = pd.DataFrame(rows)  # 存成dataframe
unformal_df.rename(columns={0: "feature", 1: "name"}, inplace=True)
rows.clear()
# 讀取formal
with open(config.formal, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        feature, name = parts[0], parts[1]
        nums = list(map(float, parts[2:]))  # 轉浮點數
        rows.append([feature, name] + nums)
formal_df = pd.DataFrame(rows)  # 存成dataframe
formal_df.rename(columns={0: "feature", 1: "name"}, inplace=True)
del rows
# 計算歐基里德距離
# unformal
unformal_numbers = unformal_df.iloc[:, 2:].to_numpy()
unformal_features = unformal_df['feature'].to_numpy()
dist_matrix = pairwise_distances(unformal_numbers, metric='euclidean')  # 計算距離
nearest_indices = np.argsort(dist_matrix, axis=1)[:, 1:11]  # 跳過自己 (index 0 是自己)
nearest_features = []
for i in range(len(unformal_df)):   # 和自己最相似的10個對應的名字和特徵
    neighbors = unformal_features[nearest_indices[i]]
    nearest_features.append(neighbors)
unformal_df["nearest_feature"] = nearest_features


# 顯示前幾列
print(unformal_df.head())