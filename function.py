from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd


def calc_ed(df):  # 計算歐基里德距離
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
    euclidean_df = pd.DataFrame(rows, columns=['feature', 'name', 'neighbors_features', 'neighbors_names'])
    return euclidean_df


def calc_cosine(df):  # Cosine Similarity
    features = df['feature'].to_numpy()  # 把df每個欄位取出來
    names = df['name'].to_numpy()
    numbers = df.iloc[:, 2:].to_numpy()
    dist_matrix = pairwise_distances(numbers, metric='cosine')
    nearest_indices = np.argsort(dist_matrix, axis=1)[:, 1:11]  # 跳過自己 (index 0 是自己)
    rows = []
    for i in range(len(df)):
        neighbors_features = features[nearest_indices[i]]
        neighbors_name = names[nearest_indices[i]]
        rows.append([features[i], names[i], neighbors_features, neighbors_name])
    cosine_df = pd.DataFrame(rows, columns=['feature', 'name', 'neighbors_features', 'neighbors_names'])
    return cosine_df


def calc_pcc(df):  # PCC
    features = df['feature'].to_numpy()
    names = df['name'].to_numpy()
    numbers = df.iloc[:, 2:].to_numpy()

    corr_matrix = 1 - squareform(pdist(numbers, metric='correlation'))  # 相似度
    nearest_indices = np.argsort(-corr_matrix, axis=1)[:, 1:11]  # 越大越相似

    rows = []
    for i in range(len(df)):
        neighbors_features = features[nearest_indices[i]]
        neighbors_names = names[nearest_indices[i]]
        rows.append([features[i], names[i], neighbors_features, neighbors_names])

    pcc_df = pd.DataFrame(rows, columns=['feature', 'name', 'neighbors_features', 'neighbors_names'])
    return pcc_df


def Percision(df):  # 計算精準度
    same_features = []
    for i in range(df.shape[0]):
        feature = df['feature'][i]
        neighbors = df['neighbors_features'][i]
        same = np.sum(neighbors == feature)
        same_features.append(same / 10)
    df['percision'] = same_features
    return df['percision'].mean()
