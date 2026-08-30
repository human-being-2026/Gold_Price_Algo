import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Price Forecasting Dashboard", page_icon=None, layout="wide")

MODEL_LABELS = {"gbr": "Gradient Boosting", "rfr": "Random Forest", "svr": "SVR", "mlp": "MLP"}
COLORS = {"gbr": "#1f77b4", "rfr": "#2ca02c", "svr": "#ff7f0e", "mlp": "#9467bd"}
PAGES = ["Overview", "Forecast", "Model Comparison", "Model Analysis", "EDA"]

# ---------------------------------------------------------------- data -----
@st.cache_data
def load_metrics():
    return pd.read_csv("results/model_metrics.csv")


@st.cache_data
def load_predictions(name):
    return pd.read_csv(f"results/predictions_{name}.csv", parse_dates=["Date"])


@st.cache_data
def load_forecast(name):
    return pd.read_csv(f"results/forecast_{name}.csv", parse_dates=["Date"])


@st.cache_data
def load_feature_importance(name):
    try:
        return pd.read_csv(f"results/feature_importance_{name}.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_cv():
    return pd.read_csv("results/cv_results.csv")


@st.cache_data
def load_eda():
    return pd.read_csv("results/eda_data.csv", parse_dates=["Date"])


@st.cache_data
def load_meta():
    with open("results/meta.json") as f:
        return json.load(f)


metrics_df = load_metrics()
meta = load_meta()
BASELINE = meta["baseline_model"]
BEST = meta["best_model"]


def styled_ax(ax):
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ------------------------------------------------------------ nav bar -----
if "page" not in st.session_state:
    st.session_state.page = "Overview"

st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] button {
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Gold Price Forecasting")
st.sidebar.caption("Machine Learning Dashboard")
for p in PAGES:
    is_active = st.session_state.page == p
    if st.sidebar.button(p, key=f"nav_{p}", width='stretch',
                          type="primary" if is_active else "secondary"):
        st.session_state.page = p
page = st.session_state.page

# ============================================================ OVERVIEW ====
if page == "Overview":
    st.title("Gold Price Forecasting System")
    st.write(
        "This dashboard uses machine learning models to forecast future gold prices "
        "based on historical market data and engineered time-series features."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Price", f"{meta['latest_price']:,.0f}")
    col2.metric("As of", meta["latest_date"])
    col3.metric("Historical Rows", f"{meta['n_rows']:,}")
    col4.metric("Models Compared", len(metrics_df))

    st.subheader("Historical Gold Price")
    hist = load_predictions("gbr")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(hist["Date"], hist["Actual"], color="#B8860B", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Gold Closing Price Over Time")
    styled_ax(ax)
    st.pyplot(fig)

    st.caption(
        f"Four regression models were trained on engineered price/return features: "
        f"Gradient Boosting, Random Forest, SVR (baseline) and MLP. See **Model Comparison** "
        f"for performance and **Forecast** for future price projections."
    )

# ============================================================ FORECAST ====
elif page == "Forecast":
    st.title("Forecast")

    c1, c2 = st.columns(2)
    with c1:
        model_key = st.selectbox("Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    with c2:
        horizon = st.slider("Forecast Horizon (trading days)", 1, meta["horizon_days"], 7)

    forecast_full = load_forecast(model_key)          # row 0 is the anchor point
    forecast_df = pd.concat([forecast_full.iloc[[0]], forecast_full.iloc[1:1 + horizon]])
    hist_df = load_predictions(model_key)

    latest_price = meta["latest_price"]
    end_price = forecast_df["Median"].iloc[-1]
    pct_change = (end_price / latest_price - 1) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest Price", f"{latest_price:,.0f}")
    m2.metric(f"Forecast (+{horizon}d, {MODEL_LABELS[model_key]})", f"{end_price:,.0f}")
    m3.metric("Expected Change", f"{pct_change:+.2f}%")

    st.subheader("Historical Price + Forecast")
    tail = hist_df.tail(180)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tail["Date"], tail["Actual"], color="#1f77b4", linewidth=1.2, label="Historical")
    ax.fill_between(forecast_df["Date"], forecast_df["Lower_5"], forecast_df["Upper_95"],
                     color="gray", alpha=0.25, label="90% Confidence Interval")
    ax.plot(forecast_df["Date"], forecast_df["Median"], color="#ff7f0e", linewidth=1.5, label="Forecast")
    ax.axvline(forecast_df["Date"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{MODEL_LABELS[model_key]}: Historical Price and {horizon}-Day Forecast")
    ax.legend(loc="upper left")
    styled_ax(ax)
    fig.autofmt_xdate()
    st.pyplot(fig)

    with st.expander("Forecast table"):
        table = forecast_df.iloc[1:].rename(columns={"Lower_5": "Lower 5%", "Upper_95": "Upper 95%"})
        st.dataframe(table, width='stretch', hide_index=True)

# ===================================================== MODEL COMPARISON ===
elif page == "Model Comparison":
    st.title("Model Comparison")

    display_df = metrics_df.copy()
    display_df["Label"] = display_df["Model"].map(MODEL_LABELS) + display_df["Baseline"].map(
        {True: " (Baseline)", False: ""}
    )

    st.dataframe(
        display_df[["Label", "Test_RMSE", "Test_MAE", "Test_R2"]]
        .rename(columns={"Label": "Model", "Test_RMSE": "RMSE", "Test_MAE": "MAE", "Test_R2": "R\u00b2"})
        .set_index("Model"),
        width='stretch',
    )

    st.subheader("Overall Comparison")
    metric_options = {"RMSE": "Test_RMSE", "MAE": "Test_MAE", "R\u00b2": "Test_R2"}
    chosen = st.multiselect("Metrics to compare", list(metric_options.keys()), default=list(metric_options.keys()))

    if chosen:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(display_df))
        n = len(chosen)
        width = 0.8 / n
        palette = ["#1f4e8c", "#2fb3c9", "#f0a500"]
        for i, metric_name in enumerate(chosen):
            col = metric_options[metric_name]
            vals = display_df[col].values
            bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
                           label=metric_name, color=palette[i % len(palette)])
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}" if metric_name == "R\u00b2" else f"{v:,.0f}",
                            (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(display_df["Label"], rotation=0)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Test-Set Metric Comparison Across Models")
        ax.legend()
        styled_ax(ax)
        st.pyplot(fig)
    else:
        st.info("Select at least one metric above to draw the chart.")

    best_row = metrics_df[metrics_df["Model"] == BEST].iloc[0]
    base_row = metrics_df[metrics_df["Model"] == BASELINE].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            f"**Baseline model: {MODEL_LABELS[BASELINE]}** \u2014 "
            f"R\u00b2 = {base_row['Test_R2']:.4f}, RMSE = {base_row['Test_RMSE']:,.0f}, "
            f"MAE = {base_row['Test_MAE']:,.0f}."
        )
    with c2:
        st.success(
            f"**Best performing model: {MODEL_LABELS[BEST]}** \u2014 "
            f"R\u00b2 = {best_row['Test_R2']:.4f}, RMSE = {best_row['Test_RMSE']:,.0f}, "
            f"MAE = {best_row['Test_MAE']:,.0f}."
        )

# ======================================================= MODEL ANALYSIS ===
elif page == "Model Analysis":
    st.title("Model Analysis")

    model_key = st.selectbox("Select Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    pred_df = load_predictions(model_key)

    st.subheader("Actual vs Predicted (full history)")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(pred_df["Date"], pred_df["Actual"], label="Actual", color="#ff7f0e", linewidth=1)
    ax.plot(pred_df["Date"], pred_df["Predicted"], label="Predicted", color="#1f77b4", linewidth=1, alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{MODEL_LABELS[model_key]}: Actual vs Predicted Next-Day Price")
    ax.legend()
    styled_ax(ax)
    st.pyplot(fig)

    st.subheader("Residuals over time")
    resid = pred_df.copy()
    resid["Residual"] = resid["Actual"] - resid["Predicted"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(resid["Date"], resid["Residual"], color="#2ca02c", linewidth=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{MODEL_LABELS[model_key]}: Prediction Residuals")
    styled_ax(ax)
    st.pyplot(fig)

    fi = load_feature_importance(model_key)
    if fi is not None:
        st.subheader("Feature Importance (Top 10)")
        top = fi.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top["Feature"], top["Importance"], color="seagreen")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"{MODEL_LABELS[model_key]}: Top 10 Feature Importances")
        styled_ax(ax)
        st.pyplot(fig)
    else:
        st.caption(f"{MODEL_LABELS[model_key]} does not expose native feature importances.")

    st.divider()
    st.subheader("Overfitting Check (Train vs Test)")
    row = metrics_df[metrics_df["Model"] == model_key].iloc[0]
    gap = row["Train_R2"] - row["Test_R2"]
    oc1, oc2, oc3 = st.columns(3)
    oc1.metric("Train R\u00b2", f"{row['Train_R2']:.4f}")
    oc2.metric("Test R\u00b2", f"{row['Test_R2']:.4f}")
    oc3.metric("R\u00b2 Gap", f"{gap:.4f}", delta=f"{-gap:.4f}", delta_color="inverse")
    if gap > 0.15:
        st.warning(
            f"Train R\u00b2 is notably higher than Test R\u00b2 (gap = {gap:.3f}). "
            "This is a sign of overfitting: the model fits the training period much "
            "better than it generalises to unseen dates."
        )
    else:
        st.caption(f"Train/Test R\u00b2 gap is {gap:.3f} \u2014 no strong overfitting signal by this measure.")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Train", "Test"], [row["Train_R2"], row["Test_R2"]], color=["#1f77b4", "#ff7f0e"])
    for b, v in zip(bars, [row["Train_R2"], row["Test_R2"]]):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center")
    ax.set_ylabel("R\u00b2")
    ax.set_xlabel("Split")
    ax.set_title(f"{MODEL_LABELS[model_key]}: Train vs Test R\u00b2")
    styled_ax(ax)
    st.pyplot(fig)

    st.subheader("5-Fold Time-Series Cross-Validation")
    st.caption(
        "Each fold trains on an earlier block of time and validates on the block right after it "
        "(TimeSeriesSplit), so no future data leaks into training. R\u00b2 is computed on the "
        "detrended training target, matching each model's own notebook."
    )
    cv_df = load_cv()
    cv_model = cv_df[cv_df["Model"] == model_key]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(cv_model["Fold"].astype(str), cv_model["R2"], color=COLORS[model_key])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("R\u00b2 (detrended target)")
    ax.set_title(f"{MODEL_LABELS[model_key]}: Cross-Validation R\u00b2 by Fold")
    styled_ax(ax)
    st.pyplot(fig)
    st.dataframe(cv_model[["Fold", "R2", "RMSE_detrended"]].set_index("Fold"), width='stretch')
    st.caption(
        f"Mean CV R\u00b2 = {cv_model['R2'].mean():.3f}. Early folds are trained on very little "
        "history and often score poorly \u2014 this is expected for a volatile, non-stationary "
        "series like daily gold price, and is worth discussing as a limitation in the report."
    )

# ================================================================== EDA ===
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    eda = load_eda()

    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(eda["Price"], bins=50, color="#B8860B", edgecolor="black", alpha=0.75)
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.set_title("Gold Price Distribution")
    styled_ax(ax)
    st.pyplot(fig)
    st.caption(
        f"Mean: {eda['Price'].mean():,.0f} | Median: {eda['Price'].median():,.0f} | "
        f"Std Dev: {eda['Price'].std():,.0f} | Skewness: {eda['Price'].skew():.2f}"
    )

    st.subheader("Price Distribution by Year")
    fig, ax = plt.subplots(figsize=(12, 5))
    years = sorted(eda["Year"].unique())
    ax.boxplot([eda.loc[eda["Year"] == y, "Price"] for y in years], tick_labels=years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    ax.set_title("Gold Price Distribution by Year")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    styled_ax(ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = ["Price", "Open", "High", "Low", "Volume", "Chg%"]
    corr = eda[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation Between Price Features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)

    st.subheader("Monthly Average % Change Heatmap")
    monthly = eda.groupby(["Year", "Month"])["Chg%"].mean().reset_index()
    pivot = monthly.pivot(index="Year", columns="Month", values="Chg%")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(pivot, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Average Monthly Gold Price Change (%)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Avg Monthly Change (%)")
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drawdown from Running Peak")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.fill_between(eda["Date"], eda["Drawdown_pct"], color="firebrick", alpha=0.4)
        ax.plot(eda["Date"], eda["Drawdown_pct"], color="firebrick", linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown From Running Peak Price")
        styled_ax(ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Indexed Price Growth")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(eda["Date"], eda["Indexed_Price"], linewidth=1.1, color="#1f4e8c")
        ax.axhline(100, linestyle="--", linewidth=1, color="gray")
        ax.set_xlabel("Date")
        ax.set_ylabel("Indexed Price (Start = 100)")
        ax.set_title("Gold Price Growth Indexed to 100")
        styled_ax(ax)
        st.pyplot(fig)

    st.subheader("Volume Distribution")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hist(eda["Volume"], bins=50, color="orange", edgecolor="black", alpha=0.75)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Frequency")
    ax.set_title("Trading Volume Distribution")
    styled_ax(ax)
    st.pyplot(fig)
