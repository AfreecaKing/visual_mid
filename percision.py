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

# 顯示前幾列
function.Percision(function.ED(formal_df))
