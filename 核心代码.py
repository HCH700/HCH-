import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

csv_file = r"D:\Users\15322\Desktop\工作簿1.csv"
out_file = r"D:\Users\15322\Desktop\blp_random_coefficients_product_data.csv"

# 推荐使用 full_market_with_positive4_interaction：
# 1. 真正的 BLP 随机系数需求估计需要保留同一市场内所有产品。
# 2. 如果只保留四个品牌，pyblp 会把其他品牌当作 outside good，市场定义会变。
# 3. 因此默认保留全市场，并加入 positive4_post_nonflagship 看四品牌异质性。
RUN_MODE = "full_market_with_positive4_interaction"
# 可选：RUN_MODE = "positive4_subset"

ADD_BRAND_FE = True
ADD_YEAR_FE = False

# 品牌扩张效应检验是品牌-月份层面的 reduced-form 检验。
# False: log_brand_quantity/share ~ post_brandfs + summer + C(brand)
# True:  额外加入 C(year)
ADD_EXPANSION_YEAR_FE = False

# 随机系数。默认只给价格加随机系数，比较稳。
# 如果要更丰富，可以改成 ["prices", "hp_num"]，但样本和工具变量压力会更大。
RANDOM_COEFFICIENTS = ["prices"]

# Monte Carlo 模拟抽样数量。先用 100 看能否收敛，正式结果可提高到 200 或 500。
INTEGRATION_SIZE = 100

target_brands = [
    "小米 Mi",
    "华凌",
    "先科 Sast",
    "新飞 Frestec"
]


def require_pyblp():
    try:
        import pyblp
        return pyblp
    except ImportError as exc:
        print("\n缺少 pyblp，随机系数 BLP 需要先安装这个包。")
        print("请在 PowerShell 运行：")
        print(r"& C:\Users\15322\AppData\Local\Programs\Python\Python313\python.exe -m pip install pyblp")
        raise SystemExit(1) from exc


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


def fit_hc1_ols(formula, data):
    model = smf.ols(formula, data=data, missing="drop")
    used_index = model.data.row_labels

    print("回归公式：", formula)
    print("回归实际样本量：", len(used_index))
    print("品牌数量：", data.loc[used_index, "brand"].nunique())
    print("标准误：HC1 稳健标准误")

    return model.fit(cov_type="HC1")


def run_one_expansion_test(agg, sample_name):
    brand_month = agg.groupby(["brand", "month", "month_num", "year"], as_index=False).agg(
        brand_quantity=("quantity", "sum"),
        market_size=("market_size", "mean"),
        first_fs_month=("first_fs_month", "min"),
        summer=("summer", "max")
    )

    brand_month["brand_share"] = brand_month["brand_quantity"] / brand_month["market_size"]

    brand_month = brand_month[
        (brand_month["brand_quantity"] > 0) &
        (brand_month["brand_share"] > 0)
    ].copy()

    brand_month["log_brand_quantity"] = np.log(brand_month["brand_quantity"])
    brand_month["log_brand_share"] = np.log(brand_month["brand_share"])
    brand_month["post_brandfs"] = (
        brand_month["first_fs_month"].notna() &
        (brand_month["month_num"] >= brand_month["first_fs_month"])
    ).astype(int)

    print(f"\n========== 品牌扩张效应检验：{sample_name} ==========")
    print("品牌-月份样本量：", len(brand_month))
    print("品牌数：", brand_month["brand"].nunique())
    print("月份数：", brand_month["month"].nunique())
    print("有进入时间的品牌数：", brand_month.loc[brand_month["first_fs_month"].notna(), "brand"].nunique())
    print("post_brandfs 分布：")
    print(brand_month["post_brandfs"].value_counts())

    if brand_month["post_brandfs"].nunique(dropna=True) < 2:
        print("post_brandfs 没有变化，无法估计品牌扩张效应。")
        return

    fixed_effects = " + C(brand)"
    if ADD_EXPANSION_YEAR_FE:
        fixed_effects += " + C(year)"

    formula_qty = "log_brand_quantity ~ post_brandfs + summer" + fixed_effects
    formula_share = "log_brand_share ~ post_brandfs + summer" + fixed_effects

    print("\n品牌总销量扩张效应：")
    expand_qty = fit_hc1_ols(formula_qty, brand_month)
    print(expand_qty.summary())

    print("\n品牌市场份额扩张效应：")
    expand_share = fit_hc1_ols(formula_share, brand_month)
    print(expand_share.summary())

    qty_coef = expand_qty.params.get("post_brandfs")
    share_coef = expand_share.params.get("post_brandfs")
    qty_pct = np.exp(qty_coef) - 1 if qty_coef is not None else np.nan
    share_pct = np.exp(share_coef) - 1 if share_coef is not None else np.nan

    print("\n扩张效应核心结果：")
    print("品牌总销量 post_brandfs 系数：", qty_coef)
    print("品牌总销量 post_brandfs p值：", expand_qty.pvalues.get("post_brandfs"))
    print("品牌总销量百分比变化：", qty_pct)
    print("品牌市场份额 post_brandfs 系数：", share_coef)
    print("品牌市场份额 post_brandfs p值：", expand_share.pvalues.get("post_brandfs"))
    print("品牌市场份额百分比变化：", share_pct)


def run_brand_expansion_tests(agg):
    print("\n========== 品牌层面：旗舰店进入后的市场扩张效应 ==========")
    print("说明：这部分是品牌-月份 reduced-form 检验，不是 BLP 结构反事实。")
    if ADD_EXPANSION_YEAR_FE:
        print("扩张检验固定效应：品牌固定效应 + 年份固定效应")
    else:
        print("扩张检验固定效应：品牌固定效应；无年份固定效应")

    run_one_expansion_test(agg, "全市场所有品牌")

    positive4 = agg[agg["brand"].isin(target_brands)].copy()
    if positive4.empty:
        print("\n四个重点品牌在样本中为空，跳过四品牌扩张检验。")
    else:
        run_one_expansion_test(positive4, "小米、华凌、先科、新飞四品牌")


def build_product_data():
    df = read_csv_auto(csv_file)

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

    print("\n进入时间解析情况：")
    print("entry_time_month 缺失行数：", df["entry_time_month"].isna().sum())
    print("entry_time_month 非缺失行数：", df["entry_time_month"].notna().sum())

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
        (df["quantity"] > 0) &
        (df["price"] > 0) &
        (df["market_size"] > 0)
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

    print("\n价格缩尾后分布：")
    print(agg["price_w"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    agg["shares"] = agg["quantity"] / agg["market_size"]

    month_share = agg.groupby("month")["shares"].sum().reset_index()
    month_share = month_share.rename(columns={"shares": "inside_share"})

    agg = agg.merge(month_share, on="month", how="left")
    agg["outside_share"] = 1 - agg["inside_share"]

    agg = agg[
        (agg["shares"] > 0) &
        (agg["outside_share"] > 0)
    ].copy()

    agg["logshare"] = np.log(agg["shares"]) - np.log(agg["outside_share"])

    agg = agg.merge(brand_entry, on="brand", how="left")

    agg["post_brandfs"] = (
        agg["first_fs_month"].notna() &
        (agg["month_num"] >= agg["first_fs_month"])
    ).astype(int)

    agg["nonflagship"] = 1 - agg["flagship"]
    agg["post_brandfs_nonflagship"] = agg["post_brandfs"] * agg["nonflagship"]
    agg["positive4_brand"] = agg["brand"].isin(target_brands).astype(int)
    agg["positive4_post_nonflagship"] = (
        agg["positive4_brand"] * agg["post_brandfs_nonflagship"]
    )

    print("\n进入后变量诊断：")
    print("post_brandfs 分布：")
    print(agg["post_brandfs"].value_counts())
    print("post_brandfs_nonflagship 分布：")
    print(agg["post_brandfs_nonflagship"].value_counts())
    print("positive4_post_nonflagship 分布：")
    print(agg["positive4_post_nonflagship"].value_counts())

    # BLP-style excluded demand instruments
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

    agg = agg.rename(columns={
        "rival_avg_hp_num": "rival_avg_hp"
    })

    expansion_base = agg.copy()

    if RUN_MODE == "positive4_subset":
        print("\n========== 四品牌子样本模式 ==========")
        print("注意：pyblp 会把被筛掉的其他品牌视为 outside good，结果仅作敏感性分析。")
        agg = agg[agg["brand"].isin(target_brands)].copy()
    elif RUN_MODE == "full_market_with_positive4_interaction":
        print("\n========== 全市场 + 四品牌交互项模式 ==========")
        print("保留所有品牌产品，并用 positive4_post_nonflagship 识别四品牌差异。")
    else:
        raise ValueError("RUN_MODE 只能是 full_market_with_positive4_interaction 或 positive4_subset。")

    if agg.empty:
        raise ValueError("样本为空，请检查 RUN_MODE 或品牌名称。")

    # pyblp 用 prices 作为价格变量名。这里除以 1000，避免价格量纲过大导致数值不稳。
    agg["prices"] = agg["price_w"] / 1000

    # pyblp 标识列
    agg["market_ids"] = agg["month"].astype(str)
    agg["firm_ids"] = agg["brand"].astype(str)
    agg["product_ids"] = (
        agg["brand"].astype(str) + "|" +
        agg["product"].astype(str) + "|" +
        agg["seller"].astype(str)
    )

    iv_vars = [
        "same_brand_product_count",
        "copper_lag3"
    ]
    iv_vars = [v for v in iv_vars if agg[v].nunique(dropna=True) > 1]

    for i, var in enumerate(iv_vars):
        agg[f"demand_instruments{i}"] = agg[var]

    needed_cols = [
        "market_ids", "firm_ids", "product_ids", "shares", "prices",
        "brand", "year", "flagship", "post_brandfs_nonflagship",
        "positive4_post_nonflagship", "hp_num", "type", "heat_cold",
        "freshair", "energysaving", "summer"
    ] + [f"demand_instruments{i}" for i in range(len(iv_vars))]

    product_data = agg[needed_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    market_share_check = product_data.groupby("market_ids")["shares"].sum()
    bad_markets = market_share_check[market_share_check >= 1].index.tolist()
    if bad_markets:
        raise ValueError(f"以下市场 inside share >= 1，无法估计 BLP：{bad_markets[:10]}")

    product_data.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("\n已导出 pyblp 产品数据：", out_file)
    print("BLP 样本量：", len(product_data))
    print("市场数：", product_data["market_ids"].nunique())
    print("品牌数：", product_data["brand"].nunique())
    print("产品数：", product_data["product_ids"].nunique())
    print("进入 BLP 的 excluded demand instruments：", iv_vars)
    print("各市场 inside share 最大值：", market_share_check.max())

    run_brand_expansion_tests(expansion_base)

    return product_data


def build_formulations(pyblp, product_data):
    linear_terms = [
        "prices",
        "flagship",
        "post_brandfs_nonflagship",
        "hp_num",
        "type",
        "heat_cold",
        "freshair",
        "energysaving",
        "summer"
    ]

    if RUN_MODE == "full_market_with_positive4_interaction":
        linear_terms.append("positive4_post_nonflagship")

    linear_terms = [
        v for v in linear_terms
        if v == "prices" or product_data[v].nunique(dropna=True) > 1
    ]

    if ADD_BRAND_FE:
        linear_terms.append("C(brand)")
    if ADD_YEAR_FE:
        linear_terms.append("C(year)")

    random_terms = [
        v for v in RANDOM_COEFFICIENTS
        if v == "prices" or product_data[v].nunique(dropna=True) > 1
    ]

    x1_formula = "1 + " + " + ".join(linear_terms)
    x2_formula = "0 + " + " + ".join(random_terms)

    print("\n========== BLP 设定 ==========")
    print("线性均值效用 X1：", x1_formula)
    print("随机系数 X2：", x2_formula)
    print("价格单位：千元，因此 prices 系数表示价格上涨 1000 元的效用变化。")

    return (
        pyblp.Formulation(x1_formula),
        pyblp.Formulation(x2_formula)
    ), random_terms


def print_core_results(results):
    print("\n========== 随机系数 BLP 核心结果 ==========")
    print("完整结果如下。重点看：")
    print("1. prices 的均值系数是否为负。")
    print("2. Sigma 中 prices 的标准差是否显著，表示价格敏感性是否存在消费者异质性。")
    print("3. post_brandfs_nonflagship 代表全样本平均同品牌非旗舰效应。")
    print("4. positive4_post_nonflagship 代表四品牌相对于其他品牌的额外差异。")
    print(results)

    for attr in ["beta", "beta_se", "sigma", "sigma_se"]:
        value = getattr(results, attr, None)
        if value is not None:
            print(f"\n{attr}:")
            print(value)


def main():
    pyblp = require_pyblp()
    pyblp.options.digits = 4
    pyblp.options.verbose = True

    product_data = build_product_data()
    product_formulations, random_terms = build_formulations(pyblp, product_data)

    integration = pyblp.Integration(
        "monte_carlo",
        size=INTEGRATION_SIZE,
        specification_options={"seed": 20260502}
    )

    problem = pyblp.Problem(
        product_formulations,
        product_data,
        integration=integration,
        add_exogenous=True
    )

    sigma_start = np.eye(len(random_terms)) * 0.5
    optimization = pyblp.Optimization(
        "l-bfgs-b",
        {"gtol": 1e-5, "maxiter": 1000}
    )

    print("\n========== 开始估计随机系数 BLP ==========")
    print("初始 Sigma：")
    print(sigma_start)

    try:
        results = problem.solve(
            sigma=sigma_start,
            method="2s",
            optimization=optimization
        )
    except Exception as exc:
        print("\n第一次估计失败，尝试用更小的 Sigma 初始值重新估计。")
        print("失败原因：", repr(exc))
        sigma_start = np.eye(len(random_terms)) * 0.1
        print("新的初始 Sigma：")
        print(sigma_start)
        results = problem.solve(
            sigma=sigma_start,
            method="2s",
            optimization=optimization
        )

    print_core_results(results)

    print("\n========== 结果解读提示 ==========")
    print("如果 Sigma[prices] 显著大于 0，说明价格敏感性存在随机异质性。")
    print("如果 Sigma[prices] 不显著，说明随机系数 BLP 没有比普通 IV logit 提供更多异质性证据。")
    print("如果 positive4_post_nonflagship 显著，说明四个品牌的非旗舰店进入后效应不同于其他品牌。")


if __name__ == "__main__":
    main()
