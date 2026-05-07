import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# DID 主设定：
# 1. 全品牌逐个扫描：log(品牌月销量) = treated + post + DID + 季节固定效应 + 误差
# 2. 自动输出全品牌 DID 系数横向图
# 3. 自动输出小米、华凌、先科、新飞四个品牌的动态 DID 图
# 4. 自动输出旗舰店进入后对品牌整体、同品牌非旗舰店的 log 销量和 log GMV 的平均影响表

CSV_FILE = Path(r"D:\Users\15322\Desktop\工作簿1.csv")
OUT_DIR = Path(r"D:\Users\15322\Desktop\DID")

TARGET_BRANDS = ["小米 Mi", "华凌", "先科 Sast", "新飞 Frestec"]
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


def month_id_from_yyyymm(yyyymm):
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def pct_from_log_coef(coef):
    return (np.exp(coef) - 1) * 100


def safe_name(name):
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(name))
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned or "brand"


def set_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def fit_clustered_ols(formula, data, cluster_col, verbose=False):
    model = smf.ols(formula, data=data, missing="drop")
    used_index = model.data.row_labels
    groups = data.loc[used_index, cluster_col]
    if verbose:
        print("回归公式：", formula)
        print("回归实际样本量：", len(used_index))
        print("聚类分组数量：", groups.nunique())
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def fit_hc1_ols(formula, data, verbose=False):
    model = smf.ols(formula, data=data, missing="drop")
    used_index = model.data.row_labels
    if verbose:
        print("回归公式：", formula)
        print("回归实际样本量：", len(used_index))
    return model.fit(cov_type="HC1")


def prepare_data():
    print("开始读取 CSV……")
    df = read_csv_auto(CSV_FILE)
    print("CSV 读入行数：", len(df))

    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(how="all")
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
        }
    )

    required_cols = ["month", "brand", "product", "entry_time", "quantity", "price"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列：{missing_cols}")

    if "flagship_raw" not in df.columns:
        df["flagship_raw"] = 0

    df["month_num"] = parse_yyyymm(df["month"])
    df["entry_time_month"] = parse_yyyymm(df["entry_time"])
    df["brand"] = df["brand"].astype("string").str.strip()
    df["product"] = df["product"].astype("string").str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["flagship_raw"] = pd.to_numeric(df["flagship_raw"], errors="coerce").fillna(0)

    df = df.dropna(subset=["brand", "product", "quantity", "price", "month_num"]).copy()
    df = df[(df["brand"] != "") & (df["product"] != "") & (df["quantity"] > 0) & (df["price"] > 0)].copy()

    df["month_num"] = df["month_num"].astype(int)
    df["year"] = df["month_num"] // 100
    df["month_in_year"] = df["month_num"] % 100
    df["season"] = ((df["month_in_year"] - 1) // 3 + 1).astype(int)
    df["month_id"] = month_id_from_yyyymm(df["month_num"])

    flagship_keys = (
        df.loc[df["entry_time_month"].notna(), ["brand", "product"]]
        .drop_duplicates()
        .assign(is_entry_flagship_store=1)
    )
    df = df.merge(flagship_keys, on=["brand", "product"], how="left")
    df["is_entry_flagship_store"] = df["is_entry_flagship_store"].fillna(0).astype(int)
    df["flagship"] = np.where(df["is_entry_flagship_store"] == 1, 1, df["flagship_raw"])
    df["flagship"] = pd.to_numeric(df["flagship"], errors="coerce").fillna(0).astype(int)
    df["nonflagship"] = 1 - df["flagship"]

    df["sales_value"] = df["quantity"] * df["price"]
    df["nonflagship_quantity_piece"] = df["quantity"] * df["nonflagship"]
    df["nonflagship_sales_value_piece"] = df["sales_value"] * df["nonflagship"]

    print("\n清洗后行数：", len(df))
    print("品牌数：", df["brand"].nunique())
    print("月份数：", df["month_num"].nunique())
    print("样本月份范围：", int(df["month_num"].min()), "到", int(df["month_num"].max()))
    print("有进入时间的旗舰店店铺数：", len(flagship_keys))
    print("有进入时间的品牌数：", flagship_keys["brand"].nunique())
    print("flagship 分布：")
    print(df["flagship"].value_counts(dropna=False))

    entry = (
        df.dropna(subset=["entry_time_month"])
        .groupby("brand", as_index=False)["entry_time_month"]
        .min()
        .rename(columns={"entry_time_month": "first_fs_month"})
    )
    entry["first_fs_month"] = entry["first_fs_month"].astype(int)
    entry["first_fs_month_id"] = month_id_from_yyyymm(entry["first_fs_month"])
    entry = entry.sort_values("first_fs_month").reset_index(drop=True)

    print("\n有旗舰店进入时间的品牌数：", len(entry))
    print(entry[["brand", "first_fs_month"]].to_string(index=False))

    brand_month = (
        df.groupby(["brand", "month_num"], as_index=False)
        .agg(
            brand_quantity=("quantity", "sum"),
            brand_gmv=("sales_value", "sum"),
            nonflagship_quantity=("nonflagship_quantity_piece", "sum"),
            nonflagship_gmv=("nonflagship_sales_value_piece", "sum"),
            year=("year", "first"),
            month_in_year=("month_in_year", "first"),
            season=("season", "first"),
            month_id=("month_id", "first"),
        )
    )

    brand_month = brand_month.merge(
        entry[["brand", "first_fs_month", "first_fs_month_id"]],
        on="brand",
        how="left",
    )
    brand_month["post_brandfs"] = (
        brand_month["first_fs_month_id"].notna()
        & (brand_month["month_id"] >= brand_month["first_fs_month_id"])
    ).astype(int)
    brand_month["has_entry_brand"] = brand_month["first_fs_month_id"].notna().astype(int)
    brand_month["target4_brand"] = brand_month["brand"].isin(TARGET_BRANDS).astype(int)
    brand_month["target4_post"] = (
        brand_month["brand"].isin(TARGET_BRANDS)
        & brand_month["first_fs_month_id"].notna()
        & (brand_month["month_id"] >= brand_month["first_fs_month_id"])
    ).astype(int)

    positive_cols = {
        "brand_quantity": "log_brand_quantity",
        "brand_gmv": "log_brand_gmv",
        "nonflagship_quantity": "log_nonflagship_quantity",
        "nonflagship_gmv": "log_nonflagship_gmv",
    }
    for raw_col, log_col in positive_cols.items():
        brand_month[log_col] = np.nan
        positive = brand_month[raw_col] > 0
        brand_month.loc[positive, log_col] = np.log(brand_month.loc[positive, raw_col])

    return brand_month, entry


def plot_coefficients(result_df, out_path):
    estimable = result_df.dropna(subset=["coef", "std_error"]).sort_values("coef")
    if estimable.empty:
        return

    set_chinese_font()

    colors = np.where(
        (estimable["coef"] > 0) & (estimable["pvalue"] < 0.05),
        "#d62728",
        np.where((estimable["coef"] < 0) & (estimable["pvalue"] < 0.05), "#1f77b4", "#8c8c8c"),
    )

    fig_height = max(7, len(estimable) * 0.35)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y = np.arange(len(estimable))

    ax.barh(y, estimable["coef"], color=colors, alpha=0.82)
    ax.errorbar(
        estimable["coef"],
        y,
        xerr=1.96 * estimable["std_error"],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.9,
        capsize=2,
        alpha=0.65,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(estimable["brand"])
    ax.set_xlabel("DID 系数：对 log(品牌总销量) 的影响")
    ax.set_title("全品牌 DID 系数：仅控制季节固定效应")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_all_brand_scan(brand_month, entry, stamp):
    formula = "log_brand_quantity ~ treated + post_policy + did_post + C(season)"
    print("\n========== 全品牌 DID：仅控制季节固定效应 ==========")
    print("回归公式：", formula)
    print("说明：本模型不加入品牌固定效应，也不加入年份固定效应。")
    print("说明：全品牌逐个扫描每次只有一个处理品牌，按品牌聚类的显著性容易偏乐观；主表采用 HC1 标准误，并同时输出品牌聚类标准误作为参考。")

    results = []
    skipped = []

    for _, row in entry.iterrows():
        brand = row["brand"]
        first_fs_month = int(row["first_fs_month"])
        policy_month_id = int(row["first_fs_month_id"])

        target = brand_month[brand_month["brand"] == brand]
        pre_months = int((target["month_id"] < policy_month_id).sum())
        post_months = int((target["month_id"] >= policy_month_id).sum())

        if pre_months == 0 or post_months == 0:
            skipped.append(
                {
                    "brand": brand,
                    "first_fs_month": first_fs_month,
                    "status": "skipped_no_pre_or_post",
                    "pre_months": pre_months,
                    "post_months": post_months,
                }
            )
            continue

        did_data = brand_month.copy()
        did_data["treated"] = (did_data["brand"] == brand).astype(int)
        did_data["post_policy"] = (did_data["month_id"] >= policy_month_id).astype(int)
        did_data["did_post"] = did_data["treated"] * did_data["post_policy"]

        try:
            res = fit_hc1_ols(formula, did_data)
            res_cluster = fit_clustered_ols(formula, did_data, "brand")
            coef = res.params.get("did_post", np.nan)
            stderr = res.bse.get("did_post", np.nan)
            pvalue = res.pvalues.get("did_post", np.nan)
            cluster_stderr = res_cluster.bse.get("did_post", np.nan)
            cluster_pvalue = res_cluster.pvalues.get("did_post", np.nan)

            results.append(
                {
                    "brand": brand,
                    "first_fs_month": first_fs_month,
                    "coef": coef,
                    "std_error": stderr,
                    "pvalue": pvalue,
                    "cluster_std_error": cluster_stderr,
                    "cluster_pvalue": cluster_pvalue,
                    "ci_low": coef - 1.96 * stderr,
                    "ci_high": coef + 1.96 * stderr,
                    "pct_change": pct_from_log_coef(coef),
                    "pre_months": pre_months,
                    "post_months": post_months,
                    "nobs": int(res.nobs),
                    "rsquared": res.rsquared,
                }
            )
        except Exception as exc:
            skipped.append(
                {
                    "brand": brand,
                    "first_fs_month": first_fs_month,
                    "status": f"failed: {exc}",
                    "pre_months": pre_months,
                    "post_months": post_months,
                }
            )

    result_df = pd.DataFrame(results)
    skipped_df = pd.DataFrame(skipped)

    raw_path = OUT_DIR / f"did_brand_season_only_raw_{stamp}.csv"
    pos_path = OUT_DIR / f"did_brand_season_only_positive_{stamp}.csv"
    neg_path = OUT_DIR / f"did_brand_season_only_negative_{stamp}.csv"
    skipped_path = OUT_DIR / f"did_brand_season_only_skipped_{stamp}.csv"
    fig_path = OUT_DIR / f"did_brand_season_only_coefficients_{stamp}.png"

    print("\n========== 扫描结果汇总 ==========")
    print("可估计品牌数：", len(result_df))
    print("跳过/失败品牌数：", len(skipped_df))

    if not result_df.empty:
        result_df = result_df.sort_values("coef", ascending=False).reset_index(drop=True)
        positive = result_df[(result_df["coef"] > 0) & (result_df["pvalue"] < 0.05)].copy()
        negative = result_df[(result_df["coef"] < 0) & (result_df["pvalue"] < 0.05)].copy()

        result_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        positive.to_csv(pos_path, index=False, encoding="utf-8-sig")
        negative.to_csv(neg_path, index=False, encoding="utf-8-sig")
        plot_coefficients(result_df, fig_path)

        print("\n正向且在 5% 水平显著的品牌数：", len(positive))
        if positive.empty:
            print("无")
        else:
            print(
                positive[
                    ["brand", "first_fs_month", "coef", "pvalue", "pct_change", "pre_months", "post_months"]
                ].to_string(index=False)
            )

        print("\n负向且在 5% 水平显著的品牌数：", len(negative))
        if negative.empty:
            print("无")
        else:
            print(
                negative.sort_values("coef")[
                    ["brand", "first_fs_month", "coef", "pvalue", "pct_change", "pre_months", "post_months"]
                ].to_string(index=False)
            )

        print("\n全量结果已导出：", raw_path)
        print("正向显著结果已导出：", pos_path)
        print("负向显著结果已导出：", neg_path)
        print("全品牌 DID 系数横向图已导出：", fig_path)

    if not skipped_df.empty:
        skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")
        print("跳过/失败品牌已导出：", skipped_path)

    return result_df, skipped_df


def build_event_terms(did_data, target_brand, policy_month_id):
    did_data = did_data.copy()
    did_data["rel_month"] = did_data["month_id"] - policy_month_id

    target_rel_months = (
        did_data.loc[
            (did_data["brand"] == target_brand)
            & did_data["rel_month"].between(-EVENT_WINDOW, EVENT_WINDOW),
            "rel_month",
        ]
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    if not target_rel_months:
        raise ValueError("事件窗口内没有该品牌的月份样本。")

    if -1 in target_rel_months:
        base_month = -1
    else:
        pre_months = [x for x in target_rel_months if x < 0]
        if not pre_months:
            raise ValueError("事件窗口内没有进入前月份，无法画动态 DID。")
        base_month = max(pre_months)

    event_terms = []
    for k in target_rel_months:
        if k == base_month:
            continue
        col = f"event_m{abs(k)}" if k < 0 else f"event_p{k}"
        did_data[col] = ((did_data["brand"] == target_brand) & (did_data["rel_month"] == k)).astype(int)
        event_terms.append((k, col))

    return did_data, event_terms, base_month


def plot_event_study(event_plot, target_brand, first_fs_month, base_month, out_path):
    if event_plot.empty:
        return

    set_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(-0.5, color="gray", linestyle="--", linewidth=1)
    ax.errorbar(
        event_plot["rel_month"],
        event_plot["coef"],
        yerr=1.96 * event_plot["std_error"],
        fmt="o-",
        capsize=4,
        color="#1f77b4",
    )
    ax.scatter([base_month], [0], color="#d62728", zorder=3, label=f"基准期 t={base_month}")
    ax.set_xlabel("相对旗舰店进入月份")
    ax.set_ylabel("动态 DID 系数：log(品牌总销量)")
    ax.set_title(f"{target_brand} 动态 DID：仅控制季节FE，进入时间 {first_fs_month}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_dynamic_did_for_targets(brand_month, entry, stamp):
    print("\n========== 四个重点品牌动态 DID ==========")
    exported = []

    for target_brand in TARGET_BRANDS:
        entry_row = entry[entry["brand"] == target_brand]
        if entry_row.empty:
            print(f"{target_brand} 没有可识别的旗舰店进入时间，跳过动态 DID。")
            continue

        first_fs_month = int(entry_row["first_fs_month"].iloc[0])
        policy_month_id = int(entry_row["first_fs_month_id"].iloc[0])

        did_data = brand_month.copy()
        did_data["treated"] = (did_data["brand"] == target_brand).astype(int)
        did_data["post_policy"] = (did_data["month_id"] >= policy_month_id).astype(int)
        did_data["did_post"] = did_data["treated"] * did_data["post_policy"]

        try:
            event_data, event_terms, base_month = build_event_terms(did_data, target_brand, policy_month_id)
            event_cols = [col for _, col in event_terms]
            event_formula = (
                "log_brand_quantity ~ treated + post_policy + "
                + " + ".join(event_cols)
                + " + C(season)"
            )
            event_res = fit_clustered_ols(event_formula, event_data, "brand")

            lead_cols = [col for k, col in event_terms if k < 0]
            pretrend_pvalue = np.nan
            if lead_cols:
                hypothesis = " = 0, ".join(lead_cols) + " = 0"
                try:
                    pretrend_test = event_res.wald_test(hypothesis, scalar=True)
                    pretrend_pvalue = float(pretrend_test.pvalue)
                except Exception as exc:
                    print(f"{target_brand} 平行趋势辅助检验失败：{exc}")

            event_rows = []
            for k, col in event_terms:
                coef_k = event_res.params.get(col, np.nan)
                se_k = event_res.bse.get(col, np.nan)
                event_rows.append(
                    {
                        "brand": target_brand,
                        "first_fs_month": first_fs_month,
                        "base_month": base_month,
                        "rel_month": k,
                        "term": col,
                        "coef": coef_k,
                        "std_error": se_k,
                        "ci_low": coef_k - 1.96 * se_k,
                        "ci_high": coef_k + 1.96 * se_k,
                        "pvalue": event_res.pvalues.get(col, np.nan),
                        "pretrend_pvalue": pretrend_pvalue,
                    }
                )

            event_plot = pd.DataFrame(event_rows).sort_values("rel_month")
            name = safe_name(target_brand)
            event_csv_path = OUT_DIR / f"did_{name}_brand_season_dynamic_{stamp}.csv"
            event_png_path = OUT_DIR / f"did_{name}_brand_season_dynamic_{stamp}.png"
            event_plot.to_csv(event_csv_path, index=False, encoding="utf-8-sig")
            plot_event_study(event_plot, target_brand, first_fs_month, base_month, event_png_path)

            exported.append(
                {
                    "brand": target_brand,
                    "first_fs_month": first_fs_month,
                    "base_month": base_month,
                    "pretrend_pvalue": pretrend_pvalue,
                    "csv": str(event_csv_path),
                    "png": str(event_png_path),
                }
            )
            print(f"{target_brand} 动态 DID 图已导出：{event_png_path}")
        except Exception as exc:
            print(f"{target_brand} 动态 DID 失败：{exc}")

    exported_df = pd.DataFrame(exported)
    if not exported_df.empty:
        exported_path = OUT_DIR / f"did_target4_dynamic_outputs_{stamp}.csv"
        exported_df.to_csv(exported_path, index=False, encoding="utf-8-sig")
        print("四品牌动态 DID 输出索引已导出：", exported_path)

    return exported_df


def run_average_effect_table(brand_month, stamp):
    print("\n========== 旗舰店进入后的平均影响表 ==========")
    print("表内回归仅控制季节固定效应，标准误按品牌聚类。")

    specs = [
        {
            "sample": "全有进入时间品牌平均",
            "group": "has_entry_brand",
            "treatment": "post_brandfs",
            "description": "所有有旗舰店进入时间的品牌，在进入后月份取 1；其他品牌/进入前月份取 0。",
        },
        {
            "sample": "小米、华凌、先科、新飞四品牌平均",
            "group": "target4_brand",
            "treatment": "target4_post",
            "description": "四个重点品牌在各自旗舰店进入后月份取 1；其他品牌和进入前月份取 0。",
        },
    ]

    outcomes = [
        {
            "level": "品牌整体",
            "outcome": "log_brand_quantity",
            "label": "log(品牌总销量)",
        },
        {
            "level": "品牌整体",
            "outcome": "log_brand_gmv",
            "label": "log(品牌总GMV)",
        },
        {
            "level": "同品牌非旗舰店",
            "outcome": "log_nonflagship_quantity",
            "label": "log(非旗舰销量)",
        },
        {
            "level": "同品牌非旗舰店",
            "outcome": "log_nonflagship_gmv",
            "label": "log(非旗舰GMV)",
        },
    ]

    rows = []
    for spec in specs:
        group = spec["group"]
        treatment = spec["treatment"]
        for outcome in outcomes:
            reg_data = brand_month.dropna(subset=[outcome["outcome"], group, treatment, "brand", "season"]).copy()
            if reg_data[treatment].nunique(dropna=True) < 2:
                rows.append(
                    {
                        "sample": spec["sample"],
                        "level": outcome["level"],
                        "outcome": outcome["label"],
                        "group": group,
                        "treatment": treatment,
                        "coef": np.nan,
                        "std_error": np.nan,
                        "pvalue": np.nan,
                        "pct_change": np.nan,
                        "nobs": len(reg_data),
                        "brand_count": reg_data["brand"].nunique(),
                        "note": "treatment 没有变化，无法估计",
                    }
                )
                continue

            formula = f"{outcome['outcome']} ~ {group} + {treatment} + C(season)"
            try:
                res = fit_clustered_ols(formula, reg_data, "brand")
                coef = res.params.get(treatment, np.nan)
                se = res.bse.get(treatment, np.nan)
                rows.append(
                    {
                        "sample": spec["sample"],
                        "level": outcome["level"],
                        "outcome": outcome["label"],
                        "group": group,
                        "treatment": treatment,
                        "coef": coef,
                        "std_error": se,
                        "pvalue": res.pvalues.get(treatment, np.nan),
                        "ci_low": coef - 1.96 * se,
                        "ci_high": coef + 1.96 * se,
                        "pct_change": pct_from_log_coef(coef),
                        "nobs": int(res.nobs),
                        "brand_count": reg_data["brand"].nunique(),
                        "note": spec["description"],
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "sample": spec["sample"],
                        "level": outcome["level"],
                        "outcome": outcome["label"],
                        "group": group,
                        "treatment": treatment,
                        "coef": np.nan,
                        "std_error": np.nan,
                        "pvalue": np.nan,
                        "pct_change": np.nan,
                        "nobs": len(reg_data),
                        "brand_count": reg_data["brand"].nunique(),
                        "note": f"估计失败：{exc}",
                    }
                )

    table = pd.DataFrame(rows)
    table_path = OUT_DIR / f"did_brand_season_average_effects_quantity_gmv_{stamp}.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    display_cols = [
        "sample",
        "level",
        "outcome",
        "coef",
        "std_error",
        "pvalue",
        "pct_change",
        "nobs",
        "brand_count",
    ]
    print(table[display_cols].to_string(index=False))
    print("平均影响表已导出：", table_path)

    return table


def run_did_outputs():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    brand_month, entry = prepare_data()

    brand_month_path = OUT_DIR / f"did_brand_season_brand_month_data_{stamp}.csv"
    brand_month.to_csv(brand_month_path, index=False, encoding="utf-8-sig")
    print("\n品牌-月份数据已导出：", brand_month_path)

    run_all_brand_scan(brand_month, entry, stamp)
    run_dynamic_did_for_targets(brand_month, entry, stamp)
    run_average_effect_table(brand_month, stamp)

    print("\n========== 解释提示 ==========")
    print("本脚本所有 DID 主设定均控制处理组主效应、post 主效应和季节固定效应；不控制品牌固定效应，也不控制年份固定效应。")
    print("全品牌横向图展示的是逐个品牌单独作为处理组时，对 log(品牌总销量) 的 DID 系数。")
    print("全品牌扫描表中的 pvalue 是 HC1 标准误结果；cluster_pvalue 是按品牌聚类的参考结果。")
    print("四品牌动态图展示进入前后每个月的相对效应，基准期通常为进入前 1 个月。")
    print("平均影响表中的 GMV = 销量 × 均价；非旗舰店结果只使用非旗舰销量/GMV 大于 0 的品牌-月份。")


if __name__ == "__main__":
    run_did_outputs()
