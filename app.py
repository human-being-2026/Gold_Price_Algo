import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Gold Price Forecasting Dashboard", page_icon=None, layout="wide")

MODEL_LABELS = {"gbr": "Gradient Boosting", "rfr": "Random Forest", "svr": "SVR", "mlp": "MLP"}
COLORS = {"gbr": "#1f77b4", "rfr": "#2ca02c", "svr": "#ff7f0e", "mlp": "#9467bd"}
PAGES = ["Overview", "Forecast", "Model Comparison", "Model Analysis", "EDA"]
TEMPLATE = "plotly_white"


def base_layout(fig, title, xaxis_title, yaxis_title, height=460):
    fig.update_layout(
        template=TEMPLATE, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


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

# ------------------------------------------------------------ nav bar -----
if "page" not in st.session_state:
    st.session_state.page = "Overview"

st.markdown(
    "<style>div[data-testid='stSidebar'] button {border-radius: 6px !important;}</style>",
    unsafe_allow_html=True,
)

st.sidebar.title("Gold Price Forecasting")
st.sidebar.caption("Machine Learning Dashboard")
for p in PAGES:
    is_active = st.session_state.page == p
    if st.sidebar.button(p, key=f"nav_{p}", width="stretch",
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
    fig = go.Figure(go.Scatter(x=hist["Date"], y=hist["Actual"], mode="lines",
                                line=dict(color="#B8860B", width=1.3), name="Price"))
    base_layout(fig, "Gold Closing Price Over Time", "Date", "Price", height=480)
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Four regression models were trained on engineered price/return features: "
        "Gradient Boosting, Random Forest, SVR (baseline) and MLP. See **Model Comparison** "
        "for performance and **Forecast** for future price projections."
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
    has_band = model_key in meta.get("monte_carlo_models", [])

    latest_price = meta["latest_price"]
    end_price = forecast_df["Path"].iloc[-1]
    pct_change = (end_price / latest_price - 1) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest Price", f"{latest_price:,.0f}")
    m2.metric(f"Forecast (+{horizon}d, {MODEL_LABELS[model_key]})", f"{end_price:,.0f}")
    m3.metric("Expected Change", f"{pct_change:+.2f}%")

    st.subheader("Historical Price + Forecast")
    tail = hist_df.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["Actual"], mode="lines",
                              line=dict(color="#1f77b4", width=1.4), name="Historical"))
    if has_band:
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_95"], mode="lines",
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_5"], mode="lines",
                                  line=dict(width=0), fill="tonexty", fillcolor="rgba(150,150,150,0.28)",
                                  name="90% Confidence Interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Path"], mode="lines",
                              line=dict(color="#ff7f0e", width=1.8), name="Forecast"))
    fig.add_vline(x=forecast_df["Date"].iloc[0], line_dash="dot", line_color="gray")
    base_layout(fig, f"{MODEL_LABELS[model_key]}: Historical Price and {horizon}-Day Forecast",
                "Date", "Price", height=520)
    st.plotly_chart(fig, width="stretch")
    if has_band:
        st.caption(
            "The shaded band is the 5th\u201395th percentile range across 150 simulated future paths "
            "(Monte Carlo + dampening, as in this model's own notebook). The orange line is one "
            "realised simulated path, so it keeps day-to-day volatility instead of showing the "
            "flat cross-simulation average."
        )
    else:
        st.caption(
            "MLP.ipynb's own forecast method is a single deterministic recursive path with one "
            "random noise draw per step \u2014 it does not run multiple simulations, so there is no "
            "Monte Carlo confidence band for this model (unlike GBR, RFR and SVR)."
        )

    with st.expander("Forecast table"):
        cols = ["Date", "Path"] + (["Lower_5", "Upper_95"] if has_band else [])
        rename = {"Path": "Forecast", "Lower_5": "Lower 5%", "Upper_95": "Upper 95%"}
        table = forecast_df.iloc[1:][cols].rename(columns=rename)
        st.dataframe(table, width="stretch", hide_index=True)

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
        width="stretch",
    )

    st.subheader("Overall Comparison")
    metric_options = {"RMSE": "Test_RMSE", "MAE": "Test_MAE", "R\u00b2": "Test_R2"}
    chosen = st.multiselect("Metrics to compare", list(metric_options.keys()), default=list(metric_options.keys()))

    if chosen:
        fig = go.Figure()
        palette = ["#1f4e8c", "#2fb3c9", "#f0a500"]
        for i, metric_name in enumerate(chosen):
            col = metric_options[metric_name]
            fig.add_trace(go.Bar(x=display_df["Label"], y=display_df[col], name=metric_name,
                                  marker_color=palette[i % len(palette)],
                                  text=[f"{v:.3f}" if metric_name == "R\u00b2" else f"{v:,.0f}"
                                        for v in display_df[col]],
                                  textposition="outside"))
        fig.update_layout(barmode="group")
        base_layout(fig, "Test-Set Metric Comparison Across Models", "Model", "Score", height=520)
        st.plotly_chart(fig, width="stretch")
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Actual"], mode="lines",
                              line=dict(color="#ff7f0e", width=1), name="Actual"))
    fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted"], mode="lines",
                              line=dict(color="#1f77b4", width=1), name="Predicted", opacity=0.85))
    base_layout(fig, f"{MODEL_LABELS[model_key]}: Actual vs Predicted Next-Day Price", "Date", "Price")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Residuals over time")
    resid = pred_df.copy()
    resid["Residual"] = resid["Actual"] - resid["Predicted"]
    fig = go.Figure(go.Scatter(x=resid["Date"], y=resid["Residual"], mode="lines",
                                line=dict(color="#2ca02c", width=0.8), name="Residual"))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    base_layout(fig, f"{MODEL_LABELS[model_key]}: Prediction Residuals", "Date",
                "Residual (Actual - Predicted)", height=350)
    st.plotly_chart(fig, width="stretch")

    fi = load_feature_importance(model_key)
    if fi is not None:
        st.subheader("Feature Importance (Top 10)")
        top = fi.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"], orientation="h",
                                marker_color="seagreen"))
        base_layout(fig, f"{MODEL_LABELS[model_key]}: Top 10 Feature Importances",
                    "Importance", "Feature", height=430)
        st.plotly_chart(fig, width="stretch")
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

    fig = go.Figure(go.Bar(x=["Train", "Test"], y=[row["Train_R2"], row["Test_R2"]],
                            marker_color=["#1f77b4", "#ff7f0e"],
                            text=[f"{row['Train_R2']:.3f}", f"{row['Test_R2']:.3f}"],
                            textposition="outside"))
    base_layout(fig, f"{MODEL_LABELS[model_key]}: Train vs Test R\u00b2", "Split", "R\u00b2", height=380)
    st.plotly_chart(fig, width="stretch")

    st.subheader("5-Fold Time-Series Cross-Validation")
    st.caption(
        "Each fold trains on an earlier block of time and validates on the block right after it "
        "(TimeSeriesSplit), so no future data leaks into training. R\u00b2 is computed on the "
        "detrended training target, matching each model's own notebook."
    )
    cv_df = load_cv()
    cv_model = cv_df[cv_df["Model"] == model_key]
    fig = go.Figure(go.Bar(x=cv_model["Fold"].astype(str), y=cv_model["R2"],
                            marker_color=COLORS[model_key]))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    base_layout(fig, f"{MODEL_LABELS[model_key]}: Cross-Validation R\u00b2 by Fold",
                "Fold", "R\u00b2 (detrended target)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cv_model[["Fold", "R2", "RMSE_detrended"]].set_index("Fold"), width="stretch")
    st.caption(
        f"Mean CV R\u00b2 = {cv_model['R2'].mean():.3f}. Early folds are trained on very little "
        "history and often score poorly \u2014 this is expected for a volatile, non-stationary "
        "series like daily gold price, and is worth discussing as a limitation in the report."
    )

# ================================================================== EDA ===
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    eda = load_eda().sort_values("Date").reset_index(drop=True)

    # --- Candlestick, last 60 trading days --------------------------------
    st.subheader("Candlestick: Last 60 Trading Days")
    recent = eda.tail(60)
    fig = go.Figure(go.Candlestick(
        x=recent["Date"], open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Price"],
        increasing_line_color="seagreen", decreasing_line_color="firebrick",
    ))
    base_layout(fig, "OHLC Candlestick \u2014 Last 60 Trading Days", "Date", "Price", height=480)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

    # --- Price distribution -------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Price Distribution")
        fig = go.Figure(go.Histogram(x=eda["Price"], nbinsx=50, marker_color="#B8860B"))
        base_layout(fig, "Gold Price Distribution", "Price", "Frequency", height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Mean: {eda['Price'].mean():,.0f} | Median: {eda['Price'].median():,.0f} | "
            f"Std Dev: {eda['Price'].std():,.0f} | Skewness: {eda['Price'].skew():.2f}"
        )
    with c2:
        st.subheader("Volume Distribution")
        fig = go.Figure(go.Histogram(x=eda["Volume"], nbinsx=50, marker_color="orange"))
        base_layout(fig, "Trading Volume Distribution", "Volume", "Frequency", height=380)
        st.plotly_chart(fig, width="stretch")

    # --- Price distribution by year (box) -------------------------------
    st.subheader("Price Distribution by Year")
    fig = px.box(eda, x="Year", y="Price", template=TEMPLATE)
    base_layout(fig, "Gold Price Distribution by Year", "Year", "Price", height=440)
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, width="stretch")

    # --- Boxplot of all numeric features ---------------------------------
    st.subheader("Boxplot of All Numerical Features")
    numeric_cols = ["Price", "Open", "High", "Low", "Volume"]
    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Box(y=eda[col], name=col))
    base_layout(fig, "Boxplot Analysis of All Numerical Features", "Feature", "Value", height=440)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # --- Correlation heatmap ----------------------------------------------
    st.subheader("Correlation Heatmap")
    numeric_cols2 = ["Price", "Open", "High", "Low", "Volume", "Chg%"]
    corr = eda[numeric_cols2].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                                text=corr.round(2).values, texttemplate="%{text}"))
    base_layout(fig, "Correlation Between Price Features", "", "", height=460)
    st.plotly_chart(fig, width="stretch")

    # --- Monthly average % change heatmap ---------------------------------
    st.subheader("Monthly Average % Change Heatmap")
    monthly = eda.groupby(["Year", "Month"])["Chg%"].mean().reset_index()
    pivot = monthly.pivot(index="Year", columns="Month", values="Chg%")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[month_names[m - 1] for m in pivot.columns],
                                y=pivot.index, colorscale="RdYlGn", zmid=0,
                                text=np.round(pivot.values, 2), texttemplate="%{text}"))
    base_layout(fig, "Average Monthly Gold Price Change (%)", "Month", "Year", height=460)
    st.plotly_chart(fig, width="stretch")

    # --- Average change by month / year -----------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Change (%) by Month")
        m_avg = eda.groupby("Month")["Chg%"].mean().reindex(range(1, 13))
        fig = go.Figure(go.Scatter(x=[month_names[m - 1] for m in m_avg.index], y=m_avg.values,
                                    mode="lines+markers", line=dict(color="#1f4e8c")))
        base_layout(fig, "Average Change (%) by Month", "Month", "Average Chg%", height=380)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Average Change (%) by Year")
        y_avg = eda.groupby("Year")["Chg%"].mean()
        fig = go.Figure(go.Scatter(x=y_avg.index, y=y_avg.values, mode="lines+markers",
                                    line=dict(color="#8c1f4e")))
        base_layout(fig, "Average Change (%) by Year", "Year", "Average Chg%", height=380)
        st.plotly_chart(fig, width="stretch")

    # --- Average High-Low difference by month ------------------------------
    st.subheader("Average High-Low Difference by Month")
    diff_avg = eda.groupby("Month")["dif"].mean().reindex(range(1, 13))
    fig = go.Figure(go.Scatter(x=[month_names[m - 1] for m in diff_avg.index], y=diff_avg.values,
                                mode="lines+markers", line=dict(color="#2fb3c9")))
    base_layout(fig, "Average High-Low Difference by Month", "Month", "Average High-Low Difference", height=380)
    st.plotly_chart(fig, width="stretch")

    # --- Scatter: Price vs Chg% ---------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Price vs Daily % Change")
        fig = go.Figure(go.Scattergl(x=eda["Price"], y=eda["Chg%"], mode="markers",
                                      marker=dict(color="#1f77b4", size=4, opacity=0.5)))
        base_layout(fig, "Gold Price vs Daily Percentage Change", "Price", "Daily Change (%)", height=420)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Distribution of Daily Changes")
        fig = go.Figure(go.Histogram(x=eda["Chg%"], nbinsx=40, marker_color="#4c72b0"))
        base_layout(fig, "Distribution of Daily Gold Price Changes", "Daily Change (%)", "Frequency", height=420)
        st.plotly_chart(fig, width="stretch")

    # --- Correlation: Chg% vs High-Low diff, with regression -----------------
    st.subheader("Correlation: |Change %| vs High-Low Difference")
    abs_chg = eda["Chg%"].abs()
    valid = eda["dif"].notna() & abs_chg.notna()
    corr_dif = np.corrcoef(eda.loc[valid, "dif"], abs_chg[valid])[0, 1]
    m, b = np.polyfit(eda.loc[valid, "dif"], abs_chg[valid], 1)
    x_line = np.linspace(eda["dif"].min(), eda["dif"].max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=eda["dif"], y=abs_chg, mode="markers",
                                marker=dict(color="#2ca02c", size=4, opacity=0.5), name="Days"))
    fig.add_trace(go.Scatter(x=x_line, y=m * x_line + b, mode="lines",
                              line=dict(color="black", width=2), name="Regression line"))
    base_layout(fig, f"Correlation Between |Chg%| and High-Low Difference (r = {corr_dif:.3f})",
                "High-Low Difference", "Absolute Change (%)", height=440)
    st.plotly_chart(fig, width="stretch")

    # --- Correlation: Chg% vs Volume, with regression -------------------------
    st.subheader("Correlation: |Change %| vs Trading Volume")
    corr_vol = np.corrcoef(eda["Volume"], abs_chg)[0, 1]
    m2, b2 = np.polyfit(eda["Volume"], abs_chg, 1)
    x_line2 = np.linspace(eda["Volume"].min(), eda["Volume"].max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=eda["Volume"], y=abs_chg, mode="markers",
                                marker=dict(color="#ff7f0e", size=4, opacity=0.5), name="Days"))
    fig.add_trace(go.Scatter(x=x_line2, y=m2 * x_line2 + b2, mode="lines",
                              line=dict(color="black", width=2), name="Regression line"))
    base_layout(fig, f"Correlation Between |Chg%| and Trade Volume (r = {corr_vol:.3f})",
                "Trade Volume", "Absolute Change (%)", height=440)
    st.plotly_chart(fig, width="stretch")

    # --- Overnight vs intraday variance share (rolling 60-day) ---------------
    st.subheader("Overnight vs Intraday Price Movement")
    roll = 60
    ov_v = eda["Overnight_Pct"].rolling(roll).var()
    id_v = eda["Intraday_Pct"].rolling(roll).var()
    share = ov_v / (ov_v + id_v) * 100
    fig = go.Figure(go.Scatter(x=eda["Date"], y=share, mode="lines", line=dict(color="#1f4e8c", width=1.3)))
    fig.add_hline(y=50, line_dash="dash", line_color="orange",
                  annotation_text="50% line: overnight = intraday", annotation_position="top left")
    base_layout(fig, "Share of 60-Day Variance From Overnight Gap", "Date",
                "Share of Variance From Overnight (%)", height=440)
    st.plotly_chart(fig, width="stretch")

    # --- Drawdown / Indexed growth -----------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drawdown from Running Peak")
        fig = go.Figure(go.Scatter(x=eda["Date"], y=eda["Drawdown_pct"], mode="lines",
                                    line=dict(color="firebrick", width=0.9),
                                    fill="tozeroy", fillcolor="rgba(178,34,34,0.3)"))
        base_layout(fig, "Drawdown From Running Peak Price", "Date", "Drawdown (%)", height=400)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Indexed Price Growth")
        fig = go.Figure(go.Scatter(x=eda["Date"], y=eda["Indexed_Price"], mode="lines",
                                    line=dict(color="#1f4e8c", width=1.1)))
        fig.add_hline(y=100, line_dash="dash", line_color="gray")
        base_layout(fig, "Gold Price Growth Indexed to 100", "Date", "Indexed Price (Start = 100)", height=400)
        st.plotly_chart(fig, width="stretch")

    # --- Annual best vs worst daily return (dumbbell) -------------------------
    st.subheader("Annual Best vs Worst Daily Change")
    annual = eda.dropna(subset=["Chg%"]).groupby("Year")["Chg%"].agg(Best="max", Worst="min").reset_index()
    fig = go.Figure()
    for _, r in annual.iterrows():
        fig.add_trace(go.Scatter(x=[r["Worst"], r["Best"]], y=[r["Year"], r["Year"]],
                                  mode="lines", line=dict(color="#A2AABB", width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=annual["Worst"], y=annual["Year"], mode="markers",
                              marker=dict(color="#C0392B", size=9), name="Worst day"))
    fig.add_trace(go.Scatter(x=annual["Best"], y=annual["Year"], mode="markers",
                              marker=dict(color="#1A6B5C", size=9), name="Best day"))
    fig.add_vline(x=0, line_color="#D0D5D8")
    base_layout(fig, "Annual Best vs Worst Daily % Change", "Daily Change (%)", "Year", height=520)
    fig.update_yaxes(type="category")
    st.plotly_chart(fig, width="stretch")
