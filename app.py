import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gold Price Forecasting Dashboard", page_icon="🥇", layout="wide")

MODEL_LABELS = {"gbr": "Gradient Boosting", "rfr": "Random Forest", "svr": "SVR", "mlp": "MLP"}


@st.cache_data
def load_metrics():
    return pd.read_csv("results/model_metrics.csv")


@st.cache_data
def load_predictions(name):
    df = pd.read_csv(f"results/predictions_{name}.csv", parse_dates=["Date"])
    return df


@st.cache_data
def load_forecast(name):
    df = pd.read_csv(f"results/forecast_{name}.csv", parse_dates=["Date"])
    return df


@st.cache_data
def load_feature_importance(name):
    try:
        return pd.read_csv(f"results/feature_importance_{name}.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_meta():
    with open("results/meta.json") as f:
        return json.load(f)


def best_model(metrics_df):
    return metrics_df.loc[metrics_df["Test_R2"].idxmax(), "Model"]


metrics_df = load_metrics()
meta = load_meta()
best = best_model(metrics_df)

st.sidebar.title("🥇 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Forecast", "Model Comparison", "Model Analysis"])

# ----------------------------------------------------------------------
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
    gbr_hist = load_predictions("gbr")
    st.line_chart(gbr_hist.set_index("Date")[["Actual"]])

    st.caption(
        "Four regression models were trained on engineered price/return features: "
        "Gradient Boosting, Random Forest, SVR and MLP. See **Model Comparison** for "
        "performance and **Forecast** for future price projections."
    )

# ----------------------------------------------------------------------
elif page == "Forecast":
    st.title("📈 Forecast")

    c1, c2 = st.columns(2)
    with c1:
        model_key = st.selectbox("Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    with c2:
        horizon = st.slider("Forecast Horizon (trading days)", 1, meta["horizon_days"], 7)

    forecast_df = load_forecast(model_key).head(horizon)
    hist_df = load_predictions(model_key)

    latest_price = meta["latest_price"]
    end_price = forecast_df["Median"].iloc[-1]
    pct_change = (end_price / latest_price - 1) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest Price", f"{latest_price:,.0f}")
    m2.metric(f"Forecast (+{horizon}d, {MODEL_LABELS[model_key]})", f"{end_price:,.0f}")
    m3.metric("Expected Change", f"{pct_change:+.2f}%")

    st.subheader("Historical Price + Forecast")
    tail = hist_df.tail(180)[["Date", "Actual"]].rename(columns={"Actual": "Historical"})
    fc = forecast_df[["Date", "Median"]].rename(columns={"Median": "Forecast"})
    combined = pd.concat([tail.set_index("Date"), fc.set_index("Date")]).sort_index()
    st.line_chart(combined)

    st.subheader("Forecast Confidence Band (5th–95th percentile)")
    band = forecast_df.set_index("Date")[["Lower_5", "Median", "Upper_95"]]
    st.line_chart(band)

    with st.expander("Forecast table"):
        st.dataframe(forecast_df, use_container_width=True)

# ----------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("🤖 Model Comparison")

    display_df = metrics_df.copy()
    display_df["Model"] = display_df["Model"].map(MODEL_LABELS)
    st.dataframe(
        display_df[["Model", "Test_RMSE", "Test_MAE", "Test_R2"]]
        .rename(columns={"Test_RMSE": "RMSE", "Test_MAE": "MAE", "Test_R2": "R²"})
        .set_index("Model"),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("RMSE")
        st.bar_chart(display_df.set_index("Model")["Test_RMSE"])
    with c2:
        st.subheader("MAE")
        st.bar_chart(display_df.set_index("Model")["Test_MAE"])
    with c3:
        st.subheader("R²")
        st.bar_chart(display_df.set_index("Model")["Test_R2"])

    best_row = metrics_df[metrics_df["Model"] == best].iloc[0]
    st.success(
        f"🏆 **Best performing model: {MODEL_LABELS[best]}** — "
        f"R² = {best_row['Test_R2']:.4f}, RMSE = {best_row['Test_RMSE']:,.0f}, "
        f"MAE = {best_row['Test_MAE']:,.0f} on the held-out test period."
    )

# ----------------------------------------------------------------------
elif page == "Model Analysis":
    st.title("🔍 Model Analysis")

    model_key = st.selectbox("Select Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    pred_df = load_predictions(model_key)

    st.subheader("Actual vs Predicted (full history)")
    st.line_chart(pred_df.set_index("Date")[["Actual", "Predicted"]])

    st.subheader("Residuals over time")
    resid = pred_df.copy()
    resid["Residual"] = resid["Actual"] - resid["Predicted"]
    st.line_chart(resid.set_index("Date")[["Residual"]])

    fi = load_feature_importance(model_key)
    if fi is not None:
        st.subheader("Feature Importance (Top 10)")
        st.bar_chart(fi.set_index("Feature")["Importance"].head(10))
    else:
        st.caption(f"{MODEL_LABELS[model_key]} does not expose native feature importances.")
