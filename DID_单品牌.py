import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


CSV_FILE = Path(r"D:\Users\15322\Desktop\工作簿1.csv")
OUT_DIR = Path(r"D:\Users\15322\Desktop")
EVENT_WINDOW = 12


def read_csv_auto(path):
    encodings = ["gbk", "gb18030", "utf-8-sig", "utf-8"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"成功读取 CSV，使用编码：{enc}")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            print(f"编码 {enc} 读取失败，尝试下一个……")
    raise last_error


def parse_yyyymm(series):
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "0": pd.NA})

    out = pd.Series(pd.NA, index=series.index, dtype="Int64")

    numeric = pd.to_numeric(s, errors="coerce")
    mask_yyyymm = numeric.between(190001, 209912)
    out.loc[mask_yyyymm] = numeric.loc[mask_yyyymm].astype("Int64")

    mask_yyyymmdd = numeric.between(19000101, 20991231)
    out.loc[mask_yyyymmdd] = (numeric.loc[mask_yyyymmdd] // 100).astype("Int64")

    remain = out.isna() & s.notna()
    if remain.any():
        dt = pd.to_datetime(s[remain], errors="coerce")
        valid = dt.notna()
        out.loc[dt.index[valid]] = (dt.loc[valid].dt.year * 100 + dt.loc[valid].dt.month).astype("Int64")

    return out


def month_id(yyyymm):
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def fit_clustered_ols(formula, data, cluster_col):
    model = smf.ols(formula, data=data, missing="drop")
    used_index = model.data.row_labels
    groups = data.loc[used_index, cluster_col]
    print("回归公式：", formula)
    print("回归实际样本量：", len(used_index))
    print("聚类分组数量：", groups.nunique())
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def pct_from_log_coef(beta):
    return (np.exp(beta) - 1) * 100


def choose_brand(available_brands):
    available_brands = sorted(str(x) for x in available_brands)
    while True:
        query = input("\n请输入要分析的品牌名（可输入完整名，也可输入关键词，如：格力、美的）：").strip()
        if not query:
            print("品牌名不能为空，请重新输入。")
            continue

        exact = [b for b in available_brands if b == query]
        if len(exact) == 1:
            return exact[0]

        query_lower = query.lower()
        matches = [b for b in available_brands if query_lower in b.lower()]
        if len(matches) == 1:
            print(f"已自动匹配品牌：{matches[0]}")
            return matches[0]

        if len(matches) > 1:
            print("匹配到多个品牌，请输入更完整的品牌名：")
            for i, brand in enumerate(matches, start=1):
                print(f"{i}. {brand}")
            continue

        print("没有找到匹配品牌。可用品牌示例：")
        for brand in available_brands[:30]:
            print(f"- {brand}")


def safe_filename_part(text):
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(text).strip())
    text = re.sub(r"\s+", "_", text)
    return text or "brand"


def with_timestamp(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_csv_safe(df, path):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = with_timestamp(path)
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"原文件可能正在被 Excel 占用，已另存为：{fallback}")
        return fallback


def savefig_safe(fig, path):
    try:
        fig.savefig(path, dpi=300)
        return path
    except PermissionError:
        fallback = with_timestamp(path)
        fig.savefig(fallback, dpi=300)
        print(f"原图片文件可能正在被占用，已另存为：{fallback}")
        return fallback


print("开始读取 CSV……")
df = read_csv_auto(CSV_FILE)

print("CSV 读入行数：", len(df))
df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
print("删除空行后行数：", len(df))
print("列名：")
print(df.columns.tolist())

df = df.rename(
    columns={
        "月": "month",
        "实际品牌": "brand",
        "产品": "product",
        "是否旗舰店": "flagship_raw",
        "进入时间": "entry_time",
        "销量(件)": "quantity",
        "均价(元)": "price",
        "市场月活": "market_size",
    }
)

required_cols = ["month", "brand", "product", "entry_time", "quantity"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"缺少必要列：{missing_cols}")

df["brand"] = df["brand"].astype("string").str.strip()
df["product"] = df["product"].astype("string").str.strip()
df["month_num"] = parse_yyyymm(df["month"])
df["entry_time_month"] = parse_yyyymm(df["entry_time"])
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

if "market_size" in df.columns:
    df["market_size"] = pd.to_numeric(df["market_size"], errors="coerce")

df = df.dropna(subset=["month_num", "brand", "product", "quantity"])
df = df[df["quantity"] > 0].copy()

df["year"] = (df["month_num"] // 100).astype(int)
df["month_in_year"] = (df["month_num"] % 100).astype(int)
df["season"] = ((df["month_in_year"] - 1) // 3 + 1).astype(int)
df["month_id"] = month_id(df["month_num"].astype(int))

print("\n清洗后行数：", len(df))
print("品牌数：", df["brand"].nunique())
print("月份数：", df["month_num"].nunique())

target_brand = choose_brand(df["brand"].dropna().unique())
file_tag = safe_filename_part(target_brand)
print(f"\n本次分析品牌：{target_brand}")

entry_by_brand = (
    df.loc[df["entry_time_month"].notna()]
    .groupby("brand", as_index=False)["entry_time_month"]
    .min()
    .rename(columns={"entry_time_month": "first_fs_month"})
)

print("\n有旗舰店进入时间的品牌数：", len(entry_by_brand))
print(entry_by_brand.sort_values("first_fs_month").head(20).to_string(index=False))

brand_entry = entry_by_brand.loc[entry_by_brand["brand"] == target_brand, "first_fs_month"]
if brand_entry.empty:
    raise ValueError(f"{target_brand} 没有旗舰店进入时间，无法定义 DID 的政策时点。")

policy_month = int(brand_entry.iloc[0])
policy_month_id = int(month_id(policy_month))
print(f"\n{target_brand} 旗舰店进入时间：{policy_month}")

brand_month = (
    df.groupby(["brand", "month_num", "year", "season", "month_id"], as_index=False)
    .agg(
        brand_quantity=("quantity", "sum"),
        market_size=("market_size", "mean") if "market_size" in df.columns else ("quantity", "sum"),
    )
)

brand_month["log_brand_quantity"] = np.log(brand_month["brand_quantity"])
brand_month["treated_brand"] = (brand_month["brand"] == target_brand).astype(int)
brand_month["post_policy"] = (brand_month["month_id"] >= policy_month_id).astype(int)
brand_month["did_post"] = brand_month["treated_brand"] * brand_month["post_policy"]
brand_month["event_time"] = brand_month["month_id"] - policy_month_id

out_csv = OUT_DIR / f"did_{file_tag}_brand_month.csv"
out_csv = write_csv_safe(brand_month, out_csv)
print("品牌-月份数据已导出：", out_csv)

print(f"\n========== {target_brand} 进入前后原始均值 ==========")
raw_summary = (
    brand_month.loc[brand_month["brand"] == target_brand]
    .groupby("post_policy")["brand_quantity"]
    .agg(["count", "mean", "median", "std"])
)
print(raw_summary)

print("\n========== Reduced-form DID：品牌总销量 ==========")
did_formula = "log_brand_quantity ~ did_post + C(brand) + C(year) + C(season)"
did_res = fit_clustered_ols(did_formula, brand_month, "brand")
print(did_res.summary())

did_beta = did_res.params.get("did_post")
did_p = did_res.pvalues.get("did_post")
did_pct = pct_from_log_coef(did_beta)
print("\n========== DID 核心结果 ==========")
print(f"DID 系数 did_post：{did_beta:.6f}")
print(f"DID p值：{did_p:.6f}")
print(f"换算为销量百分比变化：{did_pct:.2f}%")
if did_p < 0.05:
    direction = "上升" if did_beta > 0 else "下降"
    print(f"解释：{target_brand} 旗舰店进入后，品牌整体销量相对其他品牌显著{direction}。")
else:
    print(f"解释：没有证据表明 {target_brand} 旗舰店进入后，品牌整体销量相对其他品牌发生显著变化。")

print("\n========== 动态效应 / 事件研究 ==========")
event_data = brand_month.copy()

event_terms = []
for k in range(-EVENT_WINDOW, EVENT_WINDOW + 1):
    if k == -1:
        continue
    if k < 0:
        col = f"event_m{abs(k)}"
    else:
        col = f"event_p{k}"
    event_data[col] = ((event_data["treated_brand"] == 1) & (event_data["event_time"] == k)).astype(int)
    event_terms.append((k, col))

event_formula = (
    "log_brand_quantity ~ "
    + " + ".join(col for _, col in event_terms)
    + " + C(brand) + C(year) + C(season)"
)
event_res = fit_clustered_ols(event_formula, event_data, "brand")
print(event_res.summary())
print(
    f"\n说明：本事件研究只有 {target_brand} 一个处理品牌，动态系数主要作为描述性图形。"
    "单一处理组下，按品牌聚类的逐月事件系数标准误可能不稳定，"
    "显著性判断应以主 DID 回归为主。"
)

plot_rows = []
for k, col in event_terms:
    coef = event_res.params.get(col, np.nan)
    se = event_res.bse.get(col, np.nan)
    plot_rows.append(
        {
            "event_time": k,
            "coef": coef,
            "se": se,
            "ci_low": coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
            "pvalue": event_res.pvalues.get(col, np.nan),
        }
    )

event_plot = pd.DataFrame(plot_rows)
event_plot = pd.concat(
    [
        event_plot,
        pd.DataFrame(
            [
                {
                    "event_time": -1,
                    "coef": 0.0,
                    "se": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "pvalue": np.nan,
                }
            ]
        ),
    ],
    ignore_index=True,
).sort_values("event_time")

out_event_csv = OUT_DIR / f"did_{file_tag}_event_study.csv"
out_event_csv = write_csv_safe(event_plot, out_event_csv)
print("动态效应结果已导出：", out_event_csv)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 5.8))
ax.axhline(0, color="black", linewidth=1)
ax.axvline(-0.5, color="gray", linestyle="--", linewidth=1)
ax.errorbar(
    event_plot["event_time"],
    event_plot["coef"],
    yerr=[
        event_plot["coef"] - event_plot["ci_low"],
        event_plot["ci_high"] - event_plot["coef"],
    ],
    fmt="o-",
    color="#1f77b4",
    ecolor="#8fbadd",
    elinewidth=1.5,
    capsize=3,
)
ax.set_title(f"{target_brand} 旗舰店进入前后的品牌销量动态效应")
ax.set_xlabel("相对旗舰店进入时间的月份")
ax.set_ylabel("对 log(品牌总销量) 的影响")
ax.set_xticks(range(-EVENT_WINDOW, EVENT_WINDOW + 1, 2))
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

out_png = OUT_DIR / f"did_{file_tag}_dynamic_effect.png"
out_png = savefig_safe(fig, out_png)
plt.close(fig)
print("动态效应图已导出：", out_png)

print("\n========== 简要结论模板 ==========")
print(
    f"本文进一步采用 reduced-form DID 检验 {target_brand} 旗舰店进入对品牌整体销量的影响。"
    f"处理组为 {target_brand}，政策时点定义为其旗舰店进入时间 {policy_month}，"
    f"控制组为其他品牌。回归控制品牌固定效应、年份固定效应和季节固定效应，"
    f"标准误按品牌聚类。DID 系数为 {did_beta:.4f}，p 值为 {did_p:.4f}，"
    f"对应销量变化约为 {did_pct:.2f}%。"
)
