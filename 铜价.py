import pandas as pd

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOPPUSDM"

data = pd.read_csv(url)
data["observation_date"] = pd.to_datetime(data["observation_date"])

# 筛选 2019年4月至今
data = data[data["observation_date"] >= "2019-04-01"].copy()

# 改列名
data = data.rename(columns={
    "observation_date": "month",
    "PCOPPUSDM": "copper_usd_per_metric_ton"
})

# 生成滞后3个月铜价
data["copper_lag3"] = data["copper_usd_per_metric_ton"].shift(3)

# 导出
csv_file = "copper_monthly_2019_to_latest.csv"
data.to_csv(csv_file, index=False, encoding="utf-8-sig")

print(data.head())
print(data.tail())
print("已导出：", csv_file)