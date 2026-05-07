from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS, IVGMM


CSV_FILE = Path(r"D:\Users\15322\Desktop\工作簿1.csv")
OUT_DIR = Path(r"D:\Users\15322\Desktop")

# 默认按你现在的无年份固定效应 BLP-style 模型扫描。
# 如果想把年份固定效应也放回去，把这里改成 True。
USE_YEAR_FE = False

# 是否把 Hansen J p值大于 0.05 视为“未拒绝工具变量外生性”
HANSEN_ALPHA = 0.05

# 最大组合长度。设为 None 会扫描所有组合；9 个候选工具全部扫描约 383 组，会比较慢。
# 一般论文主规格不建议塞太多工具变量，先扫 1-4 个工具变量的组合更实用。
MAX_IV_COUNT = 4


def read_csv_auto(path):
    print("开始读取 CSV……")
    encodings = ["gbk", "gb18030", "utf-8-sig", "utf-8"]
    for enc in encodings:
        try:
            df0 = pd.read_csv(path, encoding=enc, low_memory=False)
            print("成功读取 CSV，使用编码：", enc)
            return df0
        except UnicodeDecodeError:
            print(f"编码 {enc} 读取失败，尝试下一个……")
    raise ValueError("CSV 读取失败。")


def parse_yyyymm(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)

    invalid = s.str.lower().isin(["", "nan", "none", "nat", "0", "0.0"])
    s = s.mask(invalid, np.nan)

    out = pd.Series(np.nan, index=series.index, dtype="float64")

    mask6 = s.str.fullmatch(r"\d{6}", na=False)
    out.loc[mask6] = pd.to_numeric(s.loc[mask6], errors="coerce")

    mask8 = s.str.fullmatch(r"\d{8}", na=False)
    out.loc[mask8] = pd.to_numeric(s.loc[mask8].str.slice(0, 6), errors="coerce")

    remain = out.isna() & s.notna()
    extracted = s.loc[remain].str.extract(r"(\d{4})\D{0,3}(\d{1,2})")
    ok = extracted[0].notna() & extracted[1].notna()

    out.loc[extracted.index[ok]] = (
        extracted.loc[ok, 0].astype(int) * 100
        + extracted.loc[ok, 1].astype(int)
    )

    month_part = out % 100
    out = out.where((month_part >= 1) & (month_part <= 12), np.nan)

    return out


def make_formula(y_var, x_vars):
    return y_var + " ~ " + " + ".join(x_vars)


def fit_clustered_ols(formula, data, cluster_col):
    model = smf.ols(formula, data=data, missing="drop")
    used_index = model.data.row_labels
    groups = data.loc[used_index, cluster_col]
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def prepare_blp_data():
    df = read_csv_auto(CSV_FILE)

    print("CSV 读入行数：", len(df))

    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(how="all")
    df = df.dropna(subset=["月", "产品"])

    print("删除空行后行数：", len(df))
    print("列名：")
    print(df.columns.tolist())

    df = df.rename(columns={
        "月": "month",
        "实际品牌": "brand",
        "产品": "product",
        "是否旗舰店": "flagship_raw",
        "进入时间": "entry_time",
        "销量(件)": "quantity",
        "均价(元)": "price",
        "匹数": "hp",
        "机型": "type",
        "冷暖型和单冷型": "heat_cold",
        "新风功能": "freshair",
        "是否为省电型": "energysaving",
        "滞后三期铜价": "copper_lag3",
        "市场月活": "market_size"
    })

    df["month_num"] = parse_yyyymm(df["month"])
    df = df.dropna(subset=["month_num"])
    df["month_num"] = df["month_num"].astype(int)
    df["month"] = df["month_num"].astype(str)
    df["year"] = df["month_num"] // 100

    df["entry_time_month"] = parse_yyyymm(df["entry_time"])

    df["month_in_year"] = df["month_num"] % 100
    df["summer"] = df["month_in_year"].isin([6, 7, 8]).astype(int)

    df["brand"] = df["brand"].astype(str).str.strip()
    df["product"] = df["product"].astype(str).str.strip()
    df["seller"] = df["product"].str.split("-", n=1).str[1].fillna("").str.strip()

    num_cols = [
        "flagship_raw", "quantity", "price",
        "type", "heat_cold", "freshair", "energysaving",
        "copper_lag3", "market_size"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    hp_map = {
        "小 1 匹": 0.8,
        "正 1 匹": 1.0,
        "大 1 匹": 1.2,
        "小 1.5 匹": 1.3,
        "正 1.5 匹": 1.5,
        "大 1.5 匹": 1.7,
        "小 2 匹": 1.8,
        "正 2 匹": 2.0,
        "大 2 匹": 2.2,
        "小 3 匹": 2.8,
        "正 3 匹": 3.0,
        "大 3 匹": 3.2,
        "4匹": 4.0,
        "5 匹": 5.0,
        "正2匹": 2.0,
        "大1.5匹": 1.7,
        "大1.5匹/大1匹": 1.45
    }

    df["hp_text"] = df["hp"].astype(str).str.strip()
    df["hp_num"] = df["hp_text"].map(hp_map)

    print("匹数无法识别的行数：", df["hp_num"].isna().sum())

    flagship_keys = (
        df.loc[df["entry_time_month"].notna(), ["brand", "product"]]
        .drop_duplicates()
    )
    flagship_keys["is_entry_flagship_store"] = 1

    df = df.merge(flagship_keys, on=["brand", "product"], how="left")
    df["is_entry_flagship_store"] = df["is_entry_flagship_store"].fillna(0).astype(int)

    df["flagship_raw"] = df["flagship_raw"].fillna(0)
    df["flagship"] = np.where(
        df["is_entry_flagship_store"] == 1,
        1,
        df["flagship_raw"]
    )
    df["flagship"] = pd.to_numeric(df["flagship"], errors="coerce").fillna(0).astype(int)

    brand_entry = (
        df.loc[df["entry_time_month"].notna()]
        .groupby("brand")["entry_time_month"]
        .min()
        .reset_index()
        .rename(columns={"entry_time_month": "first_fs_month"})
    )

    print("\n旗舰店识别情况：")
    print("有进入时间的旗舰店店铺数：", len(flagship_keys))
    print("有进入时间的品牌数：", flagship_keys["brand"].nunique())
    print("flagship 分布：")
    print(df["flagship"].value_counts(dropna=False))

    df = df.dropna(subset=[
        "month", "month_num", "year", "brand", "product", "seller",
        "quantity", "price", "market_size", "hp_num"
    ])

    df = df[
        (df["quantity"] > 0)
        & (df["price"] > 0)
        & (df["market_size"] > 0)
    ].copy()

    print("清洗后行数：", len(df))

    df["sales_value"] = df["quantity"] * df["price"]

    print("开始聚合到 产品-月份 层级……")
    agg = df.groupby(["month", "month_num", "year", "brand", "product", "seller"], as_index=False).agg(
        quantity=("quantity", "sum"),
        sales_value=("sales_value", "sum"),
        flagship=("flagship", "max"),
        entry_time_month=("entry_time_month", "min"),
        hp_num=("hp_num", "mean"),
        type=("type", "mean"),
        heat_cold=("heat_cold", "mean"),
        freshair=("freshair", "mean"),
        energysaving=("energysaving", "mean"),
        copper_lag3=("copper_lag3", "mean"),
        market_size=("market_size", "mean"),
        summer=("summer", "max")
    )
    print("聚合完成，聚合后行数：", len(agg))

    agg["price"] = agg["sales_value"] / agg["quantity"]

    print("\n价格缩尾前分布：")
    print(agg["price"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    p_low = agg["price"].quantile(0.01)
    p_high = agg["price"].quantile(0.99)
    print("价格 1% 分位数：", p_low)
    print("价格 99% 分位数：", p_high)

    agg["price_w"] = agg["price"].clip(lower=p_low, upper=p_high)
    agg["logprice"] = np.log(agg["price_w"])

    agg["share"] = agg["quantity"] / agg["market_size"]
    month_share = agg.groupby("month")["share"].sum().reset_index()
    month_share = month_share.rename(columns={"share": "inside_share"})
    agg = agg.merge(month_share, on="month", how="left")
    agg["outside_share"] = 1 - agg["inside_share"]

    agg = agg[(agg["share"] > 0) & (agg["outside_share"] > 0)].copy()
    agg["logshare"] = np.log(agg["share"]) - np.log(agg["outside_share"])

    agg = agg.merge(brand_entry, on="brand", how="left")
    agg["post_brandfs"] = (
        agg["first_fs_month"].notna()
        & (agg["month_num"] >= agg["first_fs_month"])
    ).astype(int)
    agg["nonflagship"] = 1 - agg["flagship"]
    agg["post_brandfs_nonflagship"] = agg["post_brandfs"] * agg["nonflagship"]

    print("\n进入后变量诊断：")
    print("post_brandfs 分布：")
    print(agg["post_brandfs"].value_counts())
    print("post_brandfs_nonflagship 分布：")
    print(agg["post_brandfs_nonflagship"].value_counts())

    # BLP-style 工具变量
    agg["market_product_count"] = agg.groupby("month")["product"].transform("nunique")
    agg["brand_product_count"] = agg.groupby(["month", "brand"])["product"].transform("nunique")
    agg["same_brand_product_count"] = agg["brand_product_count"] - 1
    agg["rival_product_count"] = agg["market_product_count"] - agg["brand_product_count"]

    for var in ["hp_num", "type", "freshair", "energysaving", "heat_cold"]:
        market_sum = agg.groupby("month")[var].transform("sum")
        brand_sum = agg.groupby(["month", "brand"])[var].transform("sum")
        rival_sum = market_sum - brand_sum

        agg[f"rival_avg_{var}"] = np.where(
            agg["rival_product_count"] > 0,
            rival_sum / agg["rival_product_count"],
            np.nan
        )

    agg = agg.rename(columns={"rival_avg_hp_num": "rival_avg_hp"})

    # 成本侧工具变量：滞后三期铜价。做一个月度去均值版本，避免只吃长期水平。
    # 如果不用年份固定效应，原始铜价可能包含宏观趋势；去均值版本更偏向相对成本冲击。
    agg["copper_lag3_centered"] = agg["copper_lag3"] - agg["copper_lag3"].mean()

    print("\n最终样本量：", len(agg))
    print("月份数：", agg["month"].nunique())
    print("年份数：", agg["year"].nunique())
    print("产品数：", agg["product"].nunique())
    print("品牌数：", agg["brand"].nunique())

    return agg


def get_first_stage_diag(iv2sls_res):
    out = {
        "first_stage_partial_r2": np.nan,
        "first_stage_shea_r2": np.nan,
        "first_stage_f_stat": np.nan,
        "first_stage_f_pvalue": np.nan,
        "first_stage_f_dist": "",
    }
    try:
        diag = iv2sls_res.first_stage.diagnostics
        row = diag.loc["logprice"]
        out["first_stage_partial_r2"] = row.get("partial.rsquared", np.nan)
        out["first_stage_shea_r2"] = row.get("shea.rsquared", np.nan)
        out["first_stage_f_stat"] = row.get("f.stat", np.nan)
        out["first_stage_f_pvalue"] = row.get("f.pval", np.nan)
        out["first_stage_f_dist"] = row.get("f.dist", "")
    except Exception:
        pass
    return out


def run_iv_scan():
    agg = prepare_blp_data()

    control_vars = [
        "hp_num",
        "type",
        "heat_cold",
        "freshair",
        "energysaving",
        "summer",
    ]
    control_vars = [c for c in control_vars if agg[c].nunique(dropna=True) > 1]

    effect_vars = []
    if agg["flagship"].nunique(dropna=True) > 1:
        effect_vars.append("flagship")
    if agg["post_brandfs_nonflagship"].nunique(dropna=True) > 1:
        effect_vars.append("post_brandfs_nonflagship")

    fixed_effect_vars = ["C(year)"] if USE_YEAR_FE else []
    exog_vars = effect_vars + control_vars + fixed_effect_vars

    candidate_iv_vars = [
        # 供给/竞争结构类
        "same_brand_product_count",
        "rival_product_count",
        # 竞争品特征均值
        "rival_avg_type",
        "rival_avg_hp",
        "rival_avg_freshair",
        "rival_avg_energysaving",
        "rival_avg_heat_cold",
        # 成本侧
        "copper_lag3",
        "copper_lag3_centered",
    ]
    candidate_iv_vars = [
        v for v in candidate_iv_vars
        if v in agg.columns and agg[v].nunique(dropna=True) > 1
    ]

    print("\n========== 工具变量组合扫描 ==========")
    print("是否加入年份固定效应：", USE_YEAR_FE)
    print("外生控制变量：", exog_vars)
    print("候选工具变量：", candidate_iv_vars)

    data_cols = ["logshare", "logprice", "brand"] + effect_vars + control_vars + candidate_iv_vars
    if USE_YEAR_FE:
        data_cols.append("year")

    iv_data_all = agg[data_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    print("进入 IV 扫描的样本量：", len(iv_data_all))
    print("聚类品牌数：", iv_data_all["brand"].nunique())

    all_combos = []

    for k in range(1, len(candidate_iv_vars) + 1):
        if MAX_IV_COUNT is not None and k > MAX_IV_COUNT:
            continue
        for iv_combo in combinations(candidate_iv_vars, k):
            iv_list = list(iv_combo)

            # 避免把 copper_lag3 和 copper_lag3_centered 同时放进去，二者高度共线。
            if "copper_lag3" in iv_list and "copper_lag3_centered" in iv_list:
                continue
            all_combos.append(iv_list)

    print("计划扫描组合数：", len(all_combos))
    if MAX_IV_COUNT is not None:
        print("当前最大工具变量组合长度：", MAX_IV_COUNT)
    else:
        print("当前扫描所有组合。")

    results = []

    for idx, iv_list in enumerate(all_combos, start=1):
        if idx == 1 or idx % 20 == 0 or idx == len(all_combos):
            print(f"正在估计第 {idx}/{len(all_combos)} 组：", " + ".join(iv_list))

        formula = (
            "logshare ~ 1 + "
            + " + ".join(exog_vars)
            + " + [logprice ~ "
            + " + ".join(iv_list)
            + "]"
        )

        one = {
            "iv_count": len(iv_list),
            "iv_list": " + ".join(iv_list),
            "formula": formula,
            "nobs": np.nan,
            "iv2sls_logprice_coef": np.nan,
            "iv2sls_logprice_se": np.nan,
            "iv2sls_logprice_pvalue": np.nan,
            "ivgmm_logprice_coef": np.nan,
            "ivgmm_logprice_se": np.nan,
            "ivgmm_logprice_pvalue": np.nan,
            "hansen_j_stat": np.nan,
            "hansen_j_pvalue": np.nan,
            "hansen_j_df": np.nan,
            "hansen_pass_5pct": np.nan,
            "price_negative": np.nan,
            "status": "ok",
        }

        try:
            iv2sls_res = IV2SLS.from_formula(
                formula,
                data=iv_data_all,
            ).fit(
                cov_type="clustered",
                clusters=iv_data_all["brand"],
            )

            one["nobs"] = int(iv2sls_res.nobs)
            one["iv2sls_logprice_coef"] = iv2sls_res.params.get("logprice", np.nan)
            one["iv2sls_logprice_se"] = iv2sls_res.std_errors.get("logprice", np.nan)
            one["iv2sls_logprice_pvalue"] = iv2sls_res.pvalues.get("logprice", np.nan)
            one["price_negative"] = one["iv2sls_logprice_coef"] < 0
            one.update(get_first_stage_diag(iv2sls_res))

            # Hansen J 需要过度识别：工具变量数量 > 内生变量数量。
            if len(iv_list) > 1:
                ivgmm_res = IVGMM.from_formula(
                    formula,
                    data=iv_data_all,
                    weight_type="robust",
                ).fit(
                    cov_type="clustered",
                    clusters=iv_data_all["brand"],
                )

                one["ivgmm_logprice_coef"] = ivgmm_res.params.get("logprice", np.nan)
                one["ivgmm_logprice_se"] = ivgmm_res.std_errors.get("logprice", np.nan)
                one["ivgmm_logprice_pvalue"] = ivgmm_res.pvalues.get("logprice", np.nan)
                one["hansen_j_stat"] = ivgmm_res.j_stat.stat
                one["hansen_j_pvalue"] = ivgmm_res.j_stat.pval
                one["hansen_j_df"] = ivgmm_res.j_stat.df
                one["hansen_pass_5pct"] = ivgmm_res.j_stat.pval > HANSEN_ALPHA
            else:
                one["status"] = "ok_just_identified_no_hansen"

        except Exception as exc:
            one["status"] = f"failed: {exc}"

        results.append(one)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("没有可输出的工具变量组合。")
        return

    result_df["usable_first_stage"] = result_df["first_stage_f_stat"] >= 10
    result_df["usable_price"] = (
        result_df["price_negative"].fillna(False)
        & (result_df["iv2sls_logprice_pvalue"] < 0.05)
    )
    result_df["preferred_overidentified"] = (
        (result_df["status"] == "ok")
        & result_df["usable_first_stage"].fillna(False)
        & result_df["usable_price"].fillna(False)
        & result_df["hansen_pass_5pct"].fillna(False)
    )
    result_df["preferred_just_identified"] = (
        (result_df["status"] == "ok_just_identified_no_hansen")
        & result_df["usable_first_stage"].fillna(False)
        & result_df["usable_price"].fillna(False)
    )

    result_df = result_df.sort_values(
        by=[
            "preferred_overidentified",
            "hansen_j_pvalue",
            "first_stage_f_stat",
            "iv_count",
        ],
        ascending=[False, False, False, True],
    )

    suffix = "with_year_fe" if USE_YEAR_FE else "no_year_fe"
    out_all = OUT_DIR / f"iv_instrument_scan_all_{suffix}.csv"
    out_preferred = OUT_DIR / f"iv_instrument_scan_preferred_{suffix}.csv"
    out_just = OUT_DIR / f"iv_instrument_scan_just_identified_{suffix}.csv"
    out_failed = OUT_DIR / f"iv_instrument_scan_failed_{suffix}.csv"

    result_df.to_csv(out_all, index=False, encoding="utf-8-sig")

    preferred = result_df[result_df["preferred_overidentified"]].copy()
    just = result_df[result_df["preferred_just_identified"]].copy()
    failed = result_df[result_df["status"].str.startswith("failed", na=False)].copy()

    preferred.to_csv(out_preferred, index=False, encoding="utf-8-sig")
    just.to_csv(out_just, index=False, encoding="utf-8-sig")
    failed.to_csv(out_failed, index=False, encoding="utf-8-sig")

    print("\n========== 扫描完成 ==========")
    print("全部组合数：", len(result_df))
    print("估计失败组合数：", len(failed))
    print("过度识别且通过筛选的组合数：", len(preferred))
    print("刚好识别、第一阶段强且价格显著为负的组合数：", len(just))

    show_cols = [
        "iv_count",
        "iv_list",
        "iv2sls_logprice_coef",
        "iv2sls_logprice_pvalue",
        "first_stage_f_stat",
        "first_stage_partial_r2",
        "hansen_j_pvalue",
        "hansen_pass_5pct",
        "status",
    ]

    print("\n========== 推荐：过度识别且 Hansen J 未拒绝的组合 ==========")
    if preferred.empty:
        print("没有找到同时满足：第一阶段 F>=10、价格显著为负、Hansen J p>0.05 的过度识别组合。")
    else:
        print(preferred[show_cols].head(20).to_string(index=False))

    print("\n========== 参考：刚好识别组合，无法做 Hansen J ==========")
    if just.empty:
        print("没有找到第一阶段 F>=10 且价格显著为负的刚好识别组合。")
    else:
        print(just[show_cols].head(20).to_string(index=False))

    print("\n========== Hansen J p值最高的前 20 个过度识别组合 ==========")
    overid = result_df[(result_df["status"] == "ok")].copy()
    if overid.empty:
        print("没有成功估计的过度识别组合。")
    else:
        print(
            overid.sort_values("hansen_j_pvalue", ascending=False)[show_cols]
            .head(20)
            .to_string(index=False)
        )

    print("\n全部结果已导出：", out_all)
    print("推荐过度识别组合已导出：", out_preferred)
    print("刚好识别参考组合已导出：", out_just)
    print("失败组合已导出：", out_failed)

    print("\n========== 解释提示 ==========")
    print("优先看 preferred 文件：第一阶段 F>=10、价格显著为负、Hansen J 不拒绝。")
    print("如果 preferred 为空，说明现有工具变量很难同时满足相关性和外生性。")
    print("刚好识别组合无法做 Hansen J，只能作为稳健性或参考。")
    print("如果 copper_lag3 或 copper_lag3_centered 表现更好，可以把它作为主工具变量候选。")


if __name__ == "__main__":
    run_iv_scan()
