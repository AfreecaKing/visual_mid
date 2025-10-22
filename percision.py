import config
import function
import pandas as pd

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

# 顯示結果
# 沒正規化
print("沒正規化:")
print(f"Cosine Similarity:{function.Percision(function.calc_cosine(unformal_df))}")
print(f"Euclidean Distance:{function.Percision(function.calc_ed(unformal_df))}")
print(f"PCC:{function.Percision(function.calc_pcc(unformal_df))}")
print("正規化")
print("選擇正規化方式")
number = input("1.min-max 2.z-score 3.max:")
# 讀取formal
path = ''
if number == '1':
    path = config.unformal
elif number == '2':
    path = config.z_score
elif number == '3':
    path=config.max
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        feature, name = parts[0], parts[1]
        nums = list(map(float, parts[2:]))  # 轉浮點數
        rows.append([feature, name] + nums)
formal_df = pd.DataFrame(rows)  # 存成dataframe
formal_df.rename(columns={0: "feature", 1: "name"}, inplace=True)
del rows
print(f"Cosine Similarity:{function.Percision(function.calc_cosine(formal_df))}")
print(f"Euclidean Distance:{function.Percision(function.calc_ed(formal_df))}")
print(f"PCC:{function.Percision(function.calc_pcc(formal_df))}")
