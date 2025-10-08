import numpy as np

import config
import glob
import os

os.makedirs(config.processed_data_path, exist_ok=True)  # 創建資料夾
open(config.formal, 'w').close()
open(config.unformal, 'w').close()


def z_score_normalize(features):
    # 轉成 float numpy array
    arr = np.array([float(x) for x in features], dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:  # 避免除以零
        std = 1e-8
    z_arr = (arr - mean) / std
    return z_arr.tolist()


for filename in glob.glob(config.original_data_path + "/*.txt"):  # 讀取資料夾內的txt檔
    with open(filename, 'r', encoding='utf-8') as f:
        picture_content, picture_name = [], []
        content = f.read().split()  # 用空格分開內容
        for i in range(201):  # 200 張圖
            if content[i * 482 + 480][-3:] == "jpg":  # 確定檔案是圖片
                name = content[i * 482 + 480]  # 圖片名稱讀取
                picture_name.append(name)
                picture_content.append(content[i * 482 + 1:i * 482 + 81] + content[
                    i * 482 + 337:i * 482 + 479])

    with open(config.unformal, 'a', encoding='utf-8') as f:  # 寫入資料
        for i in range(len(picture_name)):
            f.write(picture_name[i] + '\n')
            f.write(' '.join(picture_content[i]) + '\n')
    with open(config.formal, 'a', encoding='utf-8') as f:  # 寫入Z分數資料
        for i in range(len(picture_name)):
            f.write(picture_name[i] + '\n')
            z_score = z_score_normalize(picture_content[i])
            f.write(' '.join(f"{x:.4f}" for x in z_score) + '\n')
