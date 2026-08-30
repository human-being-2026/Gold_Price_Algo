"""
Train the 4 models (Gradient Boosting, Random Forest, SVR, MLP), evaluate them
on a common held-out test period, run cross-validation, and save everything
the Streamlit dashboard needs so app.py never has to retrain on page load:

    models/<name>_price_model.pkl   -> fitted "Model A" (next-day PRICE) estimator
    models/<name>_return_model.pkl  -> fitted "Model B" (next-day RETURN) estimator
    results/model_metrics.csv       -> RMSE / MAE / R2 (train+test) per model, baseline flag, normalized %
    results/cv_results.csv          -> per-fold TimeSeriesSplit CV scores per model
    results/predictions_<name>.csv  -> Date, Actual, Predicted (full history)
    results/forecast_<name>.csv     -> Date, Median, Lower_5, Upper_95 (future, anchored to last actual point)
    results/feature_importance_<name>.csv (tree models only)
    results/eda_data.csv            -> cleaned raw-ish data + derived columns for the EDA page
    results/meta.json               -> latest price/date, baseline model name, best model name

Run this once (or whenever Gold Price.csv changes):
    python train_models.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_pipeline import build_features, FEATURE_COLS

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZON_DAYS = 60      # trading days to forecast into the future
N_SIMULATIONS = 150    # Monte Carlo paths for the recursive forecast
DAMPEN_HALF_LIFE = 250
MAXLAG = 10
TEST_SIZE = 0.2
CV_FOLDS = 5
BASELINE_MODEL = "svr"  # per assignment requirement: one model must serve as the baseline

np.random.seed(42)


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def fit_log_trend(idx_train, idx_test, y_train):
    log_price_train = np.log(y_train)
    t_train = idx_train.reshape(-1, 1)
    t_test = idx_test.reshape(-1, 1)
    trend_lin = LinearRegression().fit(t_train, log_price_train)
    trend_quad = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(t_train, log_price_train)
    use_quad = r2_score(log_price_train, trend_quad.predict(t_train)) > r2_score(log_price_train, trend_lin.predict(t_train))
    trend_model = trend_quad if use_quad else trend_lin
    return trend_model, trend_model.predict(t_train), trend_model.predict(t_test)


def recursive_forecast(model, df_clean, predict_fn, horizon_days=HORIZON_DAYS,
                        n_sims=N_SIMULATIONS, residuals=None, hist_mean_return=0.0,
                        dampen_half_life=DAMPEN_HALF_LIFE, maxlag=MAXLAG, seed=7):
    """Vectorised recursive multi-step Monte Carlo forecast on Target_Return.
    The returned dataframe's FIRST row is anchored to the last actual price on
    the last actual date, so a plotted 'Historical' line and this 'Forecast'
    line share one point and connect with no visual gap."""
    last_actual_date = df_clean['Date'].max()
    last_actual_price = float(df_clean['Price'].iloc[-1])
    future_dates = pd.bdate_range(start=last_actual_date + pd.Timedelta(days=1), periods=horizon_days)

    hist_len0 = maxlag + 10
    price_hist = np.tile(df_clean['Price'].values[-hist_len0:], (n_sims, 1)).astype(float)
    return_hist = np.tile(df_clean['Return'].values[-hist_len0:], (n_sims, 1)).astype(float)
    volume_hist = np.tile(df_clean['Volume'].values[-hist_len0:], (n_sims, 1)).astype(float)
    avg_range_pct = df_clean['Daily_Range_Pct'].tail(60).mean()

    base_row = df_clean.iloc[-1][FEATURE_COLS].values.astype(float)
    feat_matrix = np.tile(base_row, (n_sims, 1))
    rng = np.random.default_rng(seed)
    all_paths = np.zeros((n_sims, horizon_days))
    col = {c: i for i, c in enumerate(FEATURE_COLS)}

    for step in range(horizon_days):
        raw_pred_returns = predict_fn(pd.DataFrame(feat_matrix, columns=FEATURE_COLS))
        dampen_weight = 0.5 ** (step / dampen_half_life)
        blended_returns = raw_pred_returns * dampen_weight + hist_mean_return * (1 - dampen_weight)
        noise = rng.choice(residuals, size=n_sims, replace=True) if residuals is not None and len(residuals) else 0.0
        combined_returns = blended_returns + noise
        new_prices = price_hist[:, -1] * (1 + combined_returns)
        all_paths[:, step] = new_prices

        fdate = future_dates[step]
        open_new = price_hist[:, -1].copy()
        half_range_pct = avg_range_pct / 2 / 100
        high_new = np.maximum(open_new * (1 + half_range_pct), new_prices)
        low_new = np.minimum(open_new * (1 - half_range_pct), new_prices)
        volume_new = volume_hist[:, -10:].mean(axis=1)

        price_hist = np.concatenate([price_hist, new_prices[:, None]], axis=1)
        return_hist = np.concatenate([return_hist, combined_returns[:, None]], axis=1)
        volume_hist = np.concatenate([volume_hist, volume_new[:, None]], axis=1)

        new_feat = np.zeros((n_sims, len(FEATURE_COLS)))
        new_feat[:, col['DayOfWeek']] = fdate.dayofweek
        new_feat[:, col['Return_Today']] = return_hist[:, -1]
        for lag in [1, 2, 3, 5, 10]:
            new_feat[:, col[f'Return_Lag_{lag}']] = return_hist[:, -lag]
        for window in [3, 5, 10, 20]:
            ma = price_hist[:, -window:].mean(axis=1)
            new_feat[:, col[f'MA_Ratio_{window}']] = new_prices / ma - 1
            new_feat[:, col[f'Vol_{window}']] = return_hist[:, -window:].std(axis=1, ddof=1)
        new_feat[:, col['Daily_Range_Pct']] = (high_new - low_new) / low_new * 100
        new_feat[:, col['Price_Open_Ratio']] = new_prices / open_new - 1
        vol_ma10 = volume_hist[:, -10:].mean(axis=1)
        new_feat[:, col['Volume_Ratio']] = volume_new / vol_ma10
        for lag in [1, 3, 5]:
            new_feat[:, col[f'Volume_Lag_{lag}']] = volume_hist[:, -lag]
        new_feat[:, col['Volume_MA_5']] = volume_hist[:, -5:].mean(axis=1)
        new_feat[:, col['Volume_MA_10']] = vol_ma10
        feat_matrix = new_feat

    p05 = np.percentile(all_paths, 5, axis=0)
    p50 = np.percentile(all_paths, 50, axis=0)
    p95 = np.percentile(all_paths, 95, axis=0)

    forecast_df = pd.DataFrame({"Date": future_dates, "Median": p50, "Lower_5": p05, "Upper_95": p95})
    anchor = pd.DataFrame({"Date": [last_actual_date], "Median": [last_actual_price],
                           "Lower_5": [last_actual_price], "Upper_95": [last_actual_price]})
    return pd.concat([anchor, forecast_df], ignore_index=True)


def save_predictions(name, dates, actual, predicted):
    pd.DataFrame({"Date": dates, "Actual": actual, "Predicted": predicted}).to_csv(
        f"{RESULTS_DIR}/predictions_{name}.csv", index=False
    )


def run_cv(estimator_factory, X_train, y_train_detrended, scale=False, n_splits=CV_FOLDS):
    """TimeSeriesSplit cross-validation on the (already detrended) training
    portion only -- mirrors what GBR.ipynb already did, extended to all models
    so overfitting / stability can be compared fairly across models."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    if scale:
        estimator = Pipeline([("scaler", StandardScaler()), ("model", estimator_factory())])
    else:
        estimator = estimator_factory()
    cv_results = cross_validate(
        estimator, X_train, y_train_detrended, cv=tscv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        return_train_score=False,
    )
    return pd.DataFrame({
        "Fold": range(1, n_splits + 1),
        "R2": cv_results["test_r2"],
        "RMSE_detrended": -cv_results["test_rmse"],
    })


def train_tree_or_kernel_model(name, estimator_factory, df_clean, scale_svr=False):
    df_A = df_clean.dropna(subset=FEATURE_COLS + ['Target']).reset_index(drop=True)
    df_A['TimeIndex'] = np.arange(len(df_A))
    X = df_A[FEATURE_COLS]
    y = df_A['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    idx_train, idx_test = train_test_split(df_A['TimeIndex'], test_size=TEST_SIZE, shuffle=False)

    trend_model, log_trend_train, log_trend_test = fit_log_trend(idx_train.values, idx_test.values, y_train.values)
    target_detrended_train = np.log(y_train.values) - log_trend_train

    # ---- 5-fold TimeSeriesSplit CV on the training portion (stability / overfitting check) ----
    cv_df = run_cv(estimator_factory, X_train, target_detrended_train, scale=scale_svr)
    cv_df.insert(0, "Model", name)
    cv_df.to_csv(f"{RESULTS_DIR}/cv_{name}.csv", index=False)

    model = estimator_factory()
    if scale_svr:
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(target_detrended_train.reshape(-1, 1))
        model.fit(scaler_X.transform(X_train), scaler_y.transform(target_detrended_train.reshape(-1, 1)).ravel())
        pred_detrended_train = scaler_y.inverse_transform(model.predict(scaler_X.transform(X_train)).reshape(-1, 1)).ravel()
        pred_detrended_test = scaler_y.inverse_transform(model.predict(scaler_X.transform(X_test)).reshape(-1, 1)).ravel()
        pred_detrended_full = scaler_y.inverse_transform(model.predict(scaler_X.transform(X)).reshape(-1, 1)).ravel()
    else:
        model.fit(X_train, target_detrended_train)
        pred_detrended_train = model.predict(X_train)
        pred_detrended_test = model.predict(X_test)
        pred_detrended_full = model.predict(X)

    log_trend_full = trend_model.predict(df_A['TimeIndex'].values.reshape(-1, 1))
    pred_price_train = np.exp(pred_detrended_train + log_trend_train)
    pred_price_test = np.exp(pred_detrended_test + log_trend_test)
    pred_price_full = np.exp(pred_detrended_full + log_trend_full)

    test_metrics = metrics(y_test.values, pred_price_test)
    train_metrics = metrics(y_train.values, pred_price_train)

    save_predictions(name, df_A['Date'], df_A['Target'], pred_price_full)

    if hasattr(model, "feature_importances_"):
        pd.DataFrame({"Feature": FEATURE_COLS, "Importance": model.feature_importances_}) \
            .sort_values("Importance", ascending=False) \
            .to_csv(f"{RESULTS_DIR}/feature_importance_{name}.csv", index=False)

    # ---- Model B: forecast engine trained on next-day RETURN ----
    df_B = df_clean.dropna(subset=['Target_Return']).reset_index(drop=True)
    X_B, y_B = df_B[FEATURE_COLS], df_B['Target_Return']
    split = int(len(X_B) * 0.8)
    X_train_B, y_train_B = X_B.iloc[:split], y_B.iloc[:split]
    X_test_B, y_test_B = X_B.iloc[split:], y_B.iloc[split:]

    forecast_model = estimator_factory()
    if scale_svr:
        scaler_X_B = StandardScaler().fit(X_B)
        scaler_y_B = StandardScaler().fit(y_B.values.reshape(-1, 1))
        eval_model = estimator_factory()
        scaler_X_eval = StandardScaler().fit(X_train_B)
        scaler_y_eval = StandardScaler().fit(y_train_B.values.reshape(-1, 1))
        eval_model.fit(scaler_X_eval.transform(X_train_B), scaler_y_eval.transform(y_train_B.values.reshape(-1, 1)).ravel())
        pred_test_B = scaler_y_eval.inverse_transform(eval_model.predict(scaler_X_eval.transform(X_test_B)).reshape(-1, 1)).ravel()
        residuals = y_test_B.values - pred_test_B
        forecast_model.fit(scaler_X_B.transform(X_B), scaler_y_B.transform(y_B.values.reshape(-1, 1)).ravel())
        predict_fn = lambda feats: scaler_y_B.inverse_transform(forecast_model.predict(scaler_X_B.transform(feats)).reshape(-1, 1)).ravel()
    else:
        eval_model = estimator_factory()
        eval_model.fit(X_train_B, y_train_B)
        residuals = y_test_B.values - eval_model.predict(X_test_B)
        forecast_model.fit(X_B, y_B)
        predict_fn = lambda feats: forecast_model.predict(feats)

    hist_mean_return = df_B['Return'].mean()
    forecast_df = recursive_forecast(
        forecast_model, df_B, predict_fn,
        residuals=residuals, hist_mean_return=hist_mean_return,
    )
    forecast_df.to_csv(f"{RESULTS_DIR}/forecast_{name}.csv", index=False)

    joblib.dump(model, f"{MODELS_DIR}/{name}_price_model.pkl")
    joblib.dump(forecast_model, f"{MODELS_DIR}/{name}_return_model.pkl")

    return {"Model": name, "Train_RMSE": train_metrics["RMSE"], "Train_MAE": train_metrics["MAE"],
            "Train_R2": train_metrics["R2"], "Test_RMSE": test_metrics["RMSE"],
            "Test_MAE": test_metrics["MAE"], "Test_R2": test_metrics["R2"]}


def train_mlp(df_raw_path="Gold Price.csv"):
    df = pd.read_csv(df_raw_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Target'] = df['Price'].shift(-1)
    df['Return'] = df['Price'].pct_change()

    for lag in [1, 2, 3, 5, 10]:
        df[f'Price_Lag_{lag}'] = df['Price'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
    for window in [3, 5, 10]:
        df[f'MA_{window}'] = df['Price'].rolling(window=window).mean()
        df[f'Std_{window}'] = df['Price'].rolling(window=window).std()
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['Price_Open_Ratio'] = df['Price'] / df['Open']

    df_clean = df.dropna().copy()
    x = np.arange(len(df_clean)).reshape(-1, 1)
    y = df_clean['Price'].values
    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(x)
    trend_model = LinearRegression().fit(x_poly, y)
    trend = trend_model.predict(x_poly)
    df_clean['Trend'] = trend
    df_clean['DPrice'] = df_clean['Price'] - df_clean['Trend']

    for lag in [1, 2, 3, 5, 10]:
        df_clean[f'Price_Lag_{lag}'] = df_clean['DPrice'].shift(lag)
    for window in [3, 5, 10]:
        df_clean[f'MA_{window}'] = df_clean['DPrice'].rolling(window=window).mean()
        df_clean[f'Std_{window}'] = df_clean['DPrice'].rolling(window=window).std()
    df_clean = df_clean.dropna().reset_index(drop=True)

    feature_cols = ['Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3',
                    'MA_3', 'MA_5', 'MA_10',
                    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5']

    f = df_clean.loc[:, df_clean.columns != 'Date']
    X = f.loc[:len(f) - 2, feature_cols]
    Y = f.loc[1:, 'DPrice']
    YPeek = f.loc[1:, 'Price']
    YT = f.loc[1:, 'Trend']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)
    dates_train, dates_test, y_trend_train, y_trend_test = train_test_split(
        df_clean['Date'][1:], YT, test_size=0.2, random_state=42, shuffle=False
    )

    # ---- 5-fold TimeSeriesSplit CV (MLP is scale-sensitive -> Pipeline with StandardScaler) ----
    cv_df = run_cv(
        lambda: MLPRegressor(hidden_layer_sizes=(11,), max_iter=500, early_stopping=True, random_state=42),
        X_train, y_train.values, scale=True,
    )
    cv_df.insert(0, "Model", "mlp")
    cv_df.to_csv(f"{RESULTS_DIR}/cv_mlp.csv", index=False)

    model = MLPRegressor(hidden_layer_sizes=(11,), max_iter=500, early_stopping=True, random_state=42)
    model.fit(X_train, y_train)

    full_pred = model.predict(X) + YT
    test_pred = model.predict(X_test) + y_trend_test
    test_actual = y_test.values + y_trend_test.values

    train_metrics = metrics(YPeek.values, full_pred.values)
    test_metrics = metrics(test_actual, test_pred.values)

    save_predictions("mlp", df_clean['Date'][1:], YPeek, full_pred)

    residuals = (test_actual - test_pred.values)

    rng = np.random.default_rng(42)
    price = list(df_clean['Price'])
    dprice = list(df_clean['DPrice'])
    retrn = list(df_clean['Return'])
    last_date = df_clean['Date'].max()
    last_price = float(price[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS)

    future_x = np.arange(len(df_clean) + HORIZON_DAYS).reshape(-1, 1)
    future_x_poly = poly.transform(future_x)
    trending_full = trend_model.predict(future_x_poly)
    future_trend = trending_full[len(df_clean):]

    sims = np.zeros((150, HORIZON_DAYS))
    for s in range(150):
        p, dp, r = list(price), list(dprice), list(retrn)
        for i in range(HORIZON_DAYS):
            row = pd.DataFrame([{
                'Price_Lag_1': dp[-1], 'Price_Lag_2': dp[-2], 'Price_Lag_3': dp[-3],
                'MA_3': np.mean(dp[-3:]), 'MA_5': np.mean(dp[-5:]), 'MA_10': np.mean(dp[-10:]),
                'Return_Lag_1': r[-1], 'Return_Lag_2': r[-2], 'Return_Lag_3': r[-3], 'Return_Lag_5': r[-5],
            }])[feature_cols]
            d = model.predict(row)[0] + rng.choice(residuals)
            n = d + future_trend[i]
            r_new = (n - p[-1]) / p[-1]
            dp.append(d); p.append(n); r.append(r_new)
            sims[s, i] = n

    p05 = np.percentile(sims, 5, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p95 = np.percentile(sims, 95, axis=0)
    forecast_df = pd.DataFrame({"Date": future_dates, "Median": p50, "Lower_5": p05, "Upper_95": p95})
    anchor = pd.DataFrame({"Date": [last_date], "Median": [last_price],
                           "Lower_5": [last_price], "Upper_95": [last_price]})
    forecast_df = pd.concat([anchor, forecast_df], ignore_index=True)
    forecast_df.to_csv(f"{RESULTS_DIR}/forecast_mlp.csv", index=False)

    joblib.dump(model, f"{MODELS_DIR}/mlp_price_model.pkl")

    return {"Model": "mlp", "Train_RMSE": train_metrics["RMSE"], "Train_MAE": train_metrics["MAE"],
            "Train_R2": train_metrics["R2"], "Test_RMSE": test_metrics["RMSE"],
            "Test_MAE": test_metrics["MAE"], "Test_R2": test_metrics["R2"]}


def build_eda_data(raw_csv_path="Gold Price.csv"):
    """Everything the EDA page needs, precomputed once so app.py never has to
    ship the raw CSV or recompute this on every reload."""
    df = pd.read_csv(raw_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month_name()
    df['dif'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['RunningMax'] = df['Price'].cummax()
    df['Drawdown_pct'] = (df['Price'] - df['RunningMax']) / df['RunningMax'] * 100
    df['Indexed_Price'] = df['Price'] / df['Price'].iloc[0] * 100

    keep_cols = ['Date', 'Year', 'Month', 'MonthName', 'Price', 'Open', 'High', 'Low',
                 'Volume', 'Chg%', 'dif', 'Daily_Range_Pct', 'Drawdown_pct', 'Indexed_Price']
    df[keep_cols].to_csv(f"{RESULTS_DIR}/eda_data.csv", index=False)


def main():
    print("Building EDA dataset ...")
    build_eda_data()

    print("Building shared features for GBR / RFR / SVR ...")
    df_clean = build_features()
    df_clean.to_csv("Gold_Price_Cleaned.csv", index=False)

    rows = []
    print("Training Gradient Boosting Regressor ...")
    rows.append(train_tree_or_kernel_model(
        "gbr",
        lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                           min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42),
        df_clean,
    ))
    print("Training Random Forest Regressor ...")
    rows.append(train_tree_or_kernel_model(
        "rfr",
        lambda: RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_split=10,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1),
        df_clean,
    ))
    print("Training Support Vector Regressor (baseline model) ...")
    rows.append(train_tree_or_kernel_model(
        "svr",
        lambda: SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'),
        df_clean,
        scale_svr=True,
    ))
    print("Training MLP Regressor ...")
    rows.append(train_mlp())

    metrics_df = pd.DataFrame(rows)
    metrics_df["Baseline"] = metrics_df["Model"] == BASELINE_MODEL

    mean_price = float(df_clean['Price'].mean())
    metrics_df["Test_RMSE_pct"] = metrics_df["Test_RMSE"] / mean_price * 100
    metrics_df["Test_MAE_pct"] = metrics_df["Test_MAE"] / mean_price * 100
    metrics_df["Test_R2_pct"] = metrics_df["Test_R2"] * 100
    metrics_df["Train_RMSE_pct"] = metrics_df["Train_RMSE"] / mean_price * 100
    metrics_df["Train_MAE_pct"] = metrics_df["Train_MAE"] / mean_price * 100
    metrics_df["Train_R2_pct"] = metrics_df["Train_R2"] * 100
    metrics_df.to_csv(f"{RESULTS_DIR}/model_metrics.csv", index=False)

    # combine per-model CV files into one table for the app
    cv_all = pd.concat([
        pd.read_csv(f"{RESULTS_DIR}/cv_{m}.csv") for m in ["gbr", "rfr", "svr", "mlp"]
    ], ignore_index=True)
    cv_all.to_csv(f"{RESULTS_DIR}/cv_results.csv", index=False)
    for m in ["gbr", "rfr", "svr", "mlp"]:
        os.remove(f"{RESULTS_DIR}/cv_{m}.csv")

    best = metrics_df.loc[metrics_df["Test_R2"].idxmax(), "Model"]

    latest_price = float(df_clean['Price'].iloc[-1])
    latest_date = str(df_clean['Date'].iloc[-1].date())
    with open(f"{RESULTS_DIR}/meta.json", "w") as f:
        json.dump({
            "latest_price": latest_price, "latest_date": latest_date,
            "n_rows": int(len(df_clean)), "horizon_days": HORIZON_DAYS,
            "baseline_model": BASELINE_MODEL, "best_model": best,
        }, f)

    print("\n=== Model comparison (test set) ===")
    print(metrics_df[["Model", "Baseline", "Test_RMSE", "Test_MAE", "Test_R2"]].to_string(index=False))
    print(f"\nBaseline model: {BASELINE_MODEL} | Best model: {best}")
    print("\n=== Cross-validation (5-fold TimeSeriesSplit, mean R2 per model) ===")
    print(cv_all.groupby("Model")["R2"].mean().to_string())
    print("\nAll artifacts saved under results/ and models/.")


if __name__ == "__main__":
    main()
