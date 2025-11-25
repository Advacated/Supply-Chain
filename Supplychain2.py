import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

# ---------------------------------------------------------
# CONFIG FLAGS – turn ON/OFF parts as you like
# ---------------------------------------------------------
DATA_PATH = "supply_chain_dataset_3M.csv"

RUN_METRICS = True
RUN_SINGLE_ECHELON_SIM = True
RUN_PLOTS = True
RUN_REPORT = True

RUN_MULTI_ECHELON_DEMO = True      # lightweight demo
RUN_RL_ENV_DEMO = True             # no training by default, just smoke test env
RUN_ML_FORECAST = True             # will train LightGBM on a sample of data

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading dataset from {path}...")
    df_local = pd.read_csv(path)
    print("Loaded:", len(df_local), "rows")
    return df_local


# ---------------------------------------------------------
# 2. METRICS: VOLATILITY & FORECAST ERRORS
# ---------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing volatility and forecast accuracy metrics...")

    df = df.copy()

    # v1
    df["error_v1"] = df["forecast_v1"] - df["true_demand"]
    df["abs_error_v1"] = df["error_v1"].abs()
    df["sq_error_v1"] = df["error_v1"] ** 2
    df["ape_v1"] = np.where(
        df["true_demand"] > 0,
        df["abs_error_v1"] / df["true_demand"],
        np.nan,
    )

    # v2
    df["error_v2"] = df["forecast_v2"] - df["true_demand"]
    df["abs_error_v2"] = df["error_v2"].abs()
    df["sq_error_v2"] = df["error_v2"] ** 2
    df["ape_v2"] = np.where(
        df["true_demand"] > 0,
        df["abs_error_v2"] / df["true_demand"],
        np.nan,
    )

    group_cols = ["product_id", "location_id"]

    metrics = df.groupby(group_cols).agg(
        mean_demand=("true_demand", "mean"),
        std_demand=("true_demand", "std"),
        mae_v1=("abs_error_v1", "mean"),
        rmse_v1=("sq_error_v1", lambda x: np.sqrt(np.mean(x))),
        mape_v1=("ape_v1", "mean"),
        mae_v2=("abs_error_v2", "mean"),
        rmse_v2=("sq_error_v2", lambda x: np.sqrt(np.mean(x))),
        mape_v2=("ape_v2", "mean"),
    )

    metrics["cv_demand"] = metrics["std_demand"] / metrics["mean_demand"]

    print("\n=== MOST VOLATILE SKUs (Top 10) ===")
    print(metrics.sort_values("cv_demand", ascending=False).head(10))
    print()

    return metrics


# ---------------------------------------------------------
# 3. SINGLE-ECHELON INVENTORY SIMULATION
# ---------------------------------------------------------

def get_sku_params(df_sku: pd.DataFrame) -> dict:
    mean_weekly = df_sku["true_demand"].mean()
    std_weekly = df_sku["true_demand"].std()
    lead_time = int(df_sku["lead_time_weeks"].iloc[0])

    mu_L = mean_weekly * lead_time
    sigma_L = std_weekly * np.sqrt(lead_time)

    return {
        "mean_weekly": mean_weekly,
        "std_weekly": std_weekly,
        "lead_time": lead_time,
        "mu_L": mu_L,
        "sigma_L": sigma_L,
        "holding_cost": df_sku["holding_cost_per_unit"].iloc[0],
        "stockout_cost": df_sku["stockout_cost_per_unit"].iloc[0],
        "expedite_cost": df_sku["expedite_cost_per_unit"].iloc[0],
    }


def simulate_sku_single_echelon(
    df_sku: pd.DataFrame,
    z: float,
    forecast_col: str = "forecast_v2",
    expedite: bool = True,
    expedite_fraction: float = 0.5,
) -> dict:
    df_sku = df_sku.sort_values("week_id").reset_index(drop=True)
    params = get_sku_params(df_sku)

    L = params["lead_time"]
    mu_L = params["mu_L"]
    sigma_L = params["sigma_L"]

    holding_cost = params["holding_cost"]
    stockout_cost = params["stockout_cost"]
    expedite_cost = params["expedite_cost"]

    # order-up-to level
    S = mu_L + z * sigma_L

    on_hand = S
    pipeline = deque()

    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    total_expedite_cost = 0.0
    total_demand = 0.0
    total_served = 0.0

    for t, row in df_sku.iterrows():
        demand = float(row["true_demand"])
        forecast = float(row[forecast_col])

        # receive
        while pipeline and pipeline[0][0] == t:
            _, qty = pipeline.popleft()
            on_hand += qty

        # demand
        if on_hand >= demand:
            served = demand
            lost = 0.0
            on_hand -= demand
        else:
            served = on_hand
            lost = demand - on_hand
            on_hand = 0.0

        # expedite
        if expedite and lost > 0:
            expedited_units = expedite_fraction * lost
            total_expedite_cost += expedited_units * expedite_cost
            served += expedited_units
            lost -= expedited_units

        # inventory position
        pipeline_qty = sum(q for _, q in pipeline)
        inv_pos = on_hand + pipeline_qty

        # simple target S
        target = S
        order_qty = max(0.0, target - inv_pos)
        if order_qty > 0:
            pipeline.append((t + L, order_qty))

        total_holding_cost += on_hand * holding_cost
        total_stockout_cost += lost * stockout_cost

        total_demand += demand
        total_served += served

    service_level = total_served / total_demand if total_demand > 0 else 1.0
    total_cost = total_holding_cost + total_stockout_cost + total_expedite_cost

    return {
        "z": z,
        "forecast": forecast_col,
        "service_level": service_level,
        "total_cost": total_cost,
        "holding_cost": total_holding_cost,
        "stockout_cost": total_stockout_cost,
        "expedite_cost": total_expedite_cost,
    }


def run_single_echelon_sim(df: pd.DataFrame, n_sample_skus: int = 50) -> pd.DataFrame:
    print("Sampling SKUs to simulate (single-echelon)...")

    unique_skus = df[["product_id", "location_id"]].drop_duplicates()
    sample_skus = unique_skus.sample(n=n_sample_skus, random_state=0)

    results = []

    for _, row in sample_skus.iterrows():
        pid, lid = row["product_id"], row["location_id"]
        df_sku = df[(df["product_id"] == pid) & (df["location_id"] == lid)]

        for z in [0.5, 1.0, 1.5, 2.0]:
            for fc in ["forecast_v1", "forecast_v2"]:
                out = simulate_sku_single_echelon(df_sku, z=z, forecast_col=fc)
                out["product_id"] = pid
                out["location_id"] = lid
                results.append(out)

    results_df = pd.DataFrame(results)

    print("\n=== POLICY SUMMARY (single-echelon) ===")
    summary = results_df.groupby(["z", "forecast"]).agg(
        avg_service=("service_level", "mean"),
        avg_cost=("total_cost", "mean"),
        avg_stockouts=("stockout_cost", "mean"),
        avg_expedite=("expedite_cost", "mean"),
    ).reset_index()

    print(summary)
    print()

    # find best z per sku with service >= 95% using forecast_v2
    print("Finding best Z per SKU (service >= 95%) ...")
    best_rows = []
    for _, sku_row in sample_skus.iterrows():
        pid, lid = sku_row["product_id"], sku_row["location_id"]
        df_sku = df[(df["product_id"] == pid) & (df["location_id"] == lid)]

        best = None
        for z in np.linspace(0.0, 3.0, 13):
            sim = simulate_sku_single_echelon(df_sku, z=z, forecast_col="forecast_v2")
            if sim["service_level"] >= 0.95:
                if best is None or sim["total_cost"] < best["total_cost"]:
                    best = {**sim, "product_id": pid, "location_id": lid}
        if best:
            best_rows.append(best)

    best_df = pd.DataFrame(best_rows)
    print("\n=== BEST Z VALUES FOR SAMPLE SKUs ===")
    if not best_df.empty:
        print(best_df[["product_id", "location_id", "z", "total_cost", "service_level"]].head())
    else:
        print("No SKUs met the service >= 95% condition in this sample.")
    print()

    return results_df, summary, best_df


# ---------------------------------------------------------
# 4. PLOTS
# ---------------------------------------------------------

def plot_cost_vs_z(summary: pd.DataFrame, output_dir: Path):
    plt.figure()
    for fc in summary["forecast"].unique():
        sub = summary[summary["forecast"] == fc]
        plt.plot(sub["z"], sub["avg_cost"], marker="o", label=fc)

    plt.xlabel("Safety Factor z")
    plt.ylabel("Average Total Cost")
    plt.title("Average Total Cost vs Safety Factor z")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = output_dir / "cost_vs_z.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_service_vs_z(summary: pd.DataFrame, output_dir: Path):
    plt.figure()
    for fc in summary["forecast"].unique():
        sub = summary[summary["forecast"] == fc]
        plt.plot(sub["z"], sub["avg_service"], marker="o", label=fc)

    plt.xlabel("Safety Factor z")
    plt.ylabel("Average Service Level")
    plt.title("Average Service Level vs Safety Factor z")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = output_dir / "service_vs_z.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_volatility_distribution(metrics: pd.DataFrame, output_dir: Path):
    plt.figure()
    plt.hist(metrics["cv_demand"].dropna(), bins=40)
    plt.xlabel("Coefficient of Variation (CV)")
    plt.ylabel("Number of SKU-Locations")
    plt.title("Distribution of Demand Volatility (CV)")
    plt.tight_layout()
    path = output_dir / "volatility_distribution.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_cost_breakout_per_sku(results_df: pd.DataFrame, output_dir: Path, z_target=2.0, fc_target="forecast_v2"):
    sub = results_df[(results_df["z"] == z_target) &
                     (results_df["forecast"] == fc_target)].copy()

    if sub.empty:
        print("No results for that z/forecast combination in cost breakout plot.")
        return

    sub_sample = sub.sample(n=min(10, len(sub)), random_state=0)

    labels = [f"{r['product_id']}-{r['location_id']}" for _, r in sub_sample.iterrows()]
    holding = sub_sample["holding_cost"].values
    stockout = sub_sample["stockout_cost"].values
    expedite = sub_sample["expedite_cost"].values

    x = np.arange(len(sub_sample))

    plt.figure(figsize=(10, 6))
    plt.bar(x, holding, label="Holding")
    plt.bar(x, stockout, bottom=holding, label="Stockout")
    plt.bar(x, expedite, bottom=holding + stockout, label="Expedite")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Cost")
    plt.title(f"Cost Breakdown per SKU (z={z_target}, {fc_target})")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "cost_breakout_per_sku.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


# ---------------------------------------------------------
# 5. REPORT GENERATION (MARKDOWN -> manual PDF)
# ---------------------------------------------------------

def write_markdown_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    best_df: pd.DataFrame,
    output_dir: Path,
):
    report_path = output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Supply Chain Simulation Report\n\n")

        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- Total rows: {len(df):,}\n")
        f.write(f"- Unique SKU-Locations: {df[['product_id','location_id']].drop_duplicates().shape[0]:,}\n\n")

        f.write("## 2. Demand Volatility\n\n")
        f.write("- Using coefficient of variation (CV = std / mean) per product-location.\n")
        f.write("- Basic stats on CV:\n\n")
        f.write(f"  - Mean CV: {metrics['cv_demand'].mean():.3f}\n")
        f.write(f"  - Median CV: {metrics['cv_demand'].median():.3f}\n")
        f.write(f"  - 90th percentile CV: {metrics['cv_demand'].quantile(0.9):.3f}\n\n")

        f.write("Top 10 most volatile SKU-locations:\n\n")
        top10 = metrics.sort_values("cv_demand", ascending=False).head(10)
        f.write(top10.to_markdown() + "\n\n")

        f.write("## 3. Policy Simulation (Single-Echelon)\n\n")
        f.write("Summary of average service and cost vs safety factor z:\n\n")
        f.write(summary.to_markdown(index=False) + "\n\n")

        f.write("## 4. Best Safety Factor per SKU (Sample)\n\n")
        if not best_df.empty:
            f.write(best_df[["product_id", "location_id", "z", "total_cost", "service_level"]]
                    .head(20).to_markdown(index=False) + "\n\n")
        else:
            f.write("No SKUs satisfied the service >= 95% constraint in this sample.\n\n")

        f.write("## 5. Plots\n\n")
        f.write("- `cost_vs_z.png`\n")
        f.write("- `service_vs_z.png`\n")
        f.write("- `volatility_distribution.png`\n")
        f.write("- `cost_breakout_per_sku.png`\n\n")

        f.write("## 6. Next Steps\n\n")
        f.write("- Refine forecasting model and re-run simulations.\n")
        f.write("- Scale up to multi-echelon and/or RL-based policies.\n")

    print(f"Markdown report written to: {report_path}")
    print("You can convert it to PDF with something like:")
    print("  pandoc report.md -o report.pdf")
    print("or export from a Markdown tool / notebook.\n")


# ---------------------------------------------------------
# 6. MULTI-ECHELON DEMO (DC -> Store)
# ---------------------------------------------------------

class Echelon:
    def __init__(self, name: str, lead_time: int, holding_cost: float, stockout_cost: float):
        self.name = name
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.on_hand = 0.0
        self.pipeline = deque()
        self.total_holding_cost = 0.0
        self.total_stockout_cost = 0.0

    def receive(self, t: int):
        while self.pipeline and self.pipeline[0][0] == t:
            _, qty = self.pipeline.popleft()
            self.on_hand += qty

    def demand_step(self, demand: float):
        if self.on_hand >= demand:
            served = demand
            lost = 0.0
            self.on_hand -= demand
        else:
            served = self.on_hand
            lost = demand - self.on_hand
            self.on_hand = 0.0

        self.total_holding_cost += self.on_hand * self.holding_cost
        self.total_stockout_cost += lost * self.stockout_cost

        return served, lost

    def order(self, t: int, target_S: float):
        inv_pos = self.on_hand + sum(q for _, q in self.pipeline)
        order_qty = max(0.0, target_S - inv_pos)
        if order_qty > 0:
            self.pipeline.append((t + self.lead_time, order_qty))
        return order_qty


def run_multi_echelon_demo(df: pd.DataFrame):
    print("\nRunning multi-echelon demo (1 SKU, DC -> Store)...")

    # pick a single SKU-location and treat it as a store
    sku = df[["product_id", "location_id"]].drop_duplicates().iloc[0]
    pid, lid = sku["product_id"], sku["location_id"]
    df_sku = df[(df["product_id"] == pid) & (df["location_id"] == lid)].sort_values("week_id")

    # parameters
    mean_weekly = df_sku["true_demand"].mean()
    std_weekly = df_sku["true_demand"].std()
    L_store = int(df_sku["lead_time_weeks"].iloc[0])
    L_dc = 2  # fixed supplier lead time
    holding_store = float(df_sku["holding_cost_per_unit"].iloc[0])
    stockout_store = float(df_sku["stockout_cost_per_unit"].iloc[0])
    holding_dc = holding_store * 0.5
    stockout_dc = stockout_store * 0.5

    store = Echelon("Store", lead_time=L_store, holding_cost=holding_store, stockout_cost=stockout_store)
    dc = Echelon("DC", lead_time=L_dc, holding_cost=holding_dc, stockout_cost=stockout_dc)

    # simple target positions
    z = 2.0
    mu_L_store = mean_weekly * L_store
    sigma_L_store = std_weekly * np.sqrt(L_store)
    S_store = mu_L_store + z * sigma_L_store

    mu_L_dc = mean_weekly * (L_dc + L_store)
    sigma_L_dc = std_weekly * np.sqrt(L_dc + L_store)
    S_dc = mu_L_dc + z * sigma_L_dc

    store.on_hand = S_store
    dc.on_hand = S_dc

    total_demand = 0.0
    total_served = 0.0

    for t, row in df_sku.iterrows():
        demand = float(row["true_demand"])
        total_demand += demand

        # DC receives from supplier
        dc.receive(t)
        # Store receives from DC
        store.receive(t)

        # customer demand hits store
        served_store, lost_store = store.demand_step(demand)
        total_served += served_store

        # store orders from DC
        store_order = store.order(t, S_store)

        # DC sees store_order as its demand
        served_dc, lost_dc = dc.demand_step(store_order)

        # DC orders from supplier
        dc.order(t, S_dc)

    service_level = total_served / total_demand if total_demand > 0 else 1.0
    total_cost = (
        store.total_holding_cost + store.total_stockout_cost +
        dc.total_holding_cost + dc.total_stockout_cost
    )

    print("Multi-echelon demo results:")
    print(f"  SKU: {pid}-{lid}")
    print(f"  Service level: {service_level:.4f}")
    print(f"  Total cost (store + DC): {total_cost:.2f}")
    print(f"  Store holding cost: {store.total_holding_cost:.2f}")
    print(f"  Store stockout cost: {store.total_stockout_cost:.2f}")
    print(f"  DC holding cost: {dc.total_holding_cost:.2f}")
    print(f"  DC stockout cost: {dc.total_stockout_cost:.2f}")
    print()


# ---------------------------------------------------------
# 7. RL ENVIRONMENT DEMO
# ---------------------------------------------------------

def rl_env_demo(df: pd.DataFrame):
    print("\nRL environment demo (no training, just a smoke test)...")
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        print("gymnasium not installed; skipping RL env demo.")
        return

    # pick a SKU
    sku = df[["product_id", "location_id"]].drop_duplicates().iloc[0]
    pid, lid = sku["product_id"], sku["location_id"]
    df_sku = df[(df["product_id"] == pid) & (df["location_id"] == lid)].sort_values("week_id").reset_index(drop=True)

    holding_cost = float(df_sku["holding_cost_per_unit"].iloc[0])
    stockout_cost = float(df_sku["stockout_cost_per_unit"].iloc[0])
    lead_time = int(df_sku["lead_time_weeks"].iloc[0])

    class InventoryEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, df_sku_inner, holding_cost_inner, stockout_cost_inner, lead_time_inner):
            super().__init__()
            self.df_sku = df_sku_inner
            self.holding_cost = holding_cost_inner
            self.stockout_cost = stockout_cost_inner
            self.lead_time = lead_time_inner
            self.T = len(df_sku_inner)

            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([1e6, 1e6]),
                dtype=np.float32,
            )

            self.action_space = spaces.Box(
                low=np.array([0.0]),
                high=np.array([1e4]),
                dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            self.on_hand = 100.0
            self.last_demand = 0.0
            self.pipeline = deque()
            obs = np.array([self.on_hand, self.last_demand], dtype=np.float32)
            return obs, {}

        def step(self, action):
            order_qty = float(action[0])

            # receive
            while self.pipeline and self.pipeline[0][0] == self.t:
                _, q = self.pipeline.popleft()
                self.on_hand += q

            demand = float(self.df_sku.loc[self.t, "true_demand"])

            if self.on_hand >= demand:
                served = demand
                lost = 0.0
                self.on_hand -= demand
            else:
                served = self.on_hand
                lost = demand - self.on_hand
                self.on_hand = 0.0

            self.pipeline.append((self.t + self.lead_time, order_qty))

            holding_cost_step = self.on_hand * self.holding_cost
            stockout_cost_step = lost * self.stockout_cost
            reward = -(holding_cost_step + stockout_cost_step)

            self.last_demand = demand
            self.t += 1
            done = self.t >= self.T

            obs = np.array([self.on_hand, self.last_demand], dtype=np.float32)
            return obs, reward, done, False, {}

    env = InventoryEnv(df_sku, holding_cost, stockout_cost, lead_time)

    # simple random policy rollout
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(10):  # just 10 steps
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"RL env demo: ran {env.t} steps, cumulative reward: {total_reward:.2f}")
    print("You can now plug this env into stable-baselines3 PPO/DQN, etc.\n")


# ---------------------------------------------------------
# 8. LIGHTGBM FORECASTING MODEL
# ---------------------------------------------------------

def run_ml_forecast(df: pd.DataFrame):
    print("\nRunning LightGBM forecasting model on a sample...")

    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except ImportError:
        print("lightgbm or scikit-learn not installed; skipping ML forecast.")
        return

    df_ml = df.copy()

    # time feature
    df_ml["week_of_year"] = (df_ml["week_id"] - 1) % 52

    # sort & create lagged demand per SKU-location
    df_ml = df_ml.sort_values(["product_id", "location_id", "week_id"])
    for lag in [1, 2, 3, 4]:
        df_ml[f"lag_{lag}"] = df_ml.groupby(["product_id", "location_id"])["true_demand"].shift(lag)

    df_ml = df_ml.dropna(subset=["lag_1", "lag_2", "lag_3", "lag_4"])

    feature_cols = [
        "forecast_v1",
        "forecast_v2",
        "promotion_flag",
        "holiday_flag",
        "week_of_year",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
    ]

    # sample subset to keep training light
    df_ml_sample = df_ml.sample(n=min(200_000, len(df_ml)), random_state=0)

    X = df_ml_sample[feature_cols]
    y = df_ml_sample["true_demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = np.mean(np.abs((y_test - y_pred) / y_test))

    print("LightGBM forecast results:")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAPE: {mape:.3f}")
    print()

    return model


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)

    metrics = None
    if RUN_METRICS:
        metrics = compute_metrics(df)
    else:
        metrics = None

    results_df = None
    summary = None
    best_df = None
    if RUN_SINGLE_ECHELON_SIM:
        results_df, summary, best_df = run_single_echelon_sim(df)
    else:
        results_df, summary, best_df = None, None, None

    if RUN_PLOTS and metrics is not None and summary is not None and results_df is not None:
        plot_volatility_distribution(metrics, output_dir)
        plot_cost_vs_z(summary, output_dir)
        plot_service_vs_z(summary, output_dir)
        plot_cost_breakout_per_sku(results_df, output_dir)

    if RUN_REPORT and metrics is not None and summary is not None:
        if best_df is None:
            best_df = pd.DataFrame()
        write_markdown_report(df, metrics, summary, best_df, output_dir)

    if RUN_MULTI_ECHELON_DEMO:
        run_multi_echelon_demo(df)

    if RUN_RL_ENV_DEMO:
        rl_env_demo(df)

    if RUN_ML_FORECAST:
        run_ml_forecast(df)

    print("All done.")


if __name__ == "__main__":
    main()