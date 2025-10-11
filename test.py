import numpy as np
import config
import glob
import os

os.makedirs(config.processed_data_path, exist_ok=True)
open(config.z_score, 'w').close()
open(config.unformal, 'w').close()
open(config.max, 'w').close()

for filename in glob.glob(config.original_data_path + "/*.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        picture_content, picture_name = [], []
        content = f.read().split()
        for i in range(201):
            if content[i * 482 + 480][-3:] == "jpg":
                name = content[i * 482 + 480]
                picture_name.append(name)
                picture_content.append(content[i * 482 + 1:i * 482 + 81] +
                                       content[i * 482 + 337:i * 482 + 479])

        data = np.array(picture_content, dtype=float)

        # 整個資料集的 mean 和 std
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            std = 1e-8
        # 整個資料集的max
        max_abs = np.max(np.abs(data))
        if max_abs == 0:
            max_abs = 1e-8
        # 正規化
        z_normalized = (data - mean) / std
        max_normalized = data / max_abs

    # 寫入未正規化資料
    with open(config.unformal, 'a', encoding='utf-8') as f:
        for i in range(len(picture_name)):
            f.write(picture_name[i][:-8] + ' ')
            f.write(picture_name[i] + ' ')
            f.write(' '.join(picture_content[i]) + '\n')

    # 寫入 Z-score 正規化資料
    with open(config.z_score, 'a', encoding='utf-8') as f:
        for i in range(len(picture_name)):
            f.write(picture_name[i][:-8] + ' ')
            f.write(picture_name[i] + ' ')
            f.write(' '.join(f"{x:.4f}" for x in z_normalized[i]) + '\n')
    # 寫入 max 正規化資料
    with open(config.max, 'a', encoding='utf-8') as f:
        for i in range(len(picture_name)):
            f.write(picture_name[i][:-8] + ' ')
            f.write(picture_name[i] + ' ')
            f.write(' '.join(f"{x:.4f}" for x in max_normalized[i]) + '\n')


