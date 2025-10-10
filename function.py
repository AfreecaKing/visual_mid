from itertools import count

from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd


def ED(df):  # 計算歐基里德距離
    features = df['feature'].to_numpy()  # 把df每個欄位取出來
    names = df['name'].to_numpy()
    numbers = df.iloc[:, 2:].to_numpy()
    dist_matrix = pairwise_distances(numbers, metric='euclidean')
    nearest_indices = np.argsort(dist_matrix, axis=1)[:, 1:11]  # 跳過自己 (index 0 是自己)
    rows = []
    for i in range(len(df)):
        neighbors_features = features[nearest_indices[i]]
        neighbors_name = names[nearest_indices[i]]
        rows.append([features[i], names[i], neighbors_features, neighbors_name])
    euclidean_df = pd.DataFrame(rows, columns=['feature', 'name', 'neighbors_features', 'neighbors_name'])
    return euclidean_df


def Percision(df):  # 計算精準度
    same_features = []
    for i in range(df.shape[0]):
        feature = df['feature'][i]
        neighbors = df['neighbors_features'][i]
        same = np.sum(neighbors == feature)
        same_features.append(same / 10)
    df['percision'] = same_features
    print(df['percision'].mean())