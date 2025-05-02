import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------------------------------------------------------
# Data retrieval
# -----------------------------------------------------------------------------

def get_stock_data(stock_code: str, start_date: str = "2000-01-01") -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance via *yfinance*.

    Columns are flattened to single level and converted to lower‑case so that
    later code can safely assume the presence of "open", "high", "low",
    "close", "adj close" and "volume".
    """
    df = yf.download(stock_code, start=start_date, auto_adjust=False)

    # Flatten any multi‑level column index returned by yfinance
    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(level=1)

    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    print("Stock data retrieved successfully. Here are the first 5 rows:")
    print(df.head())
    return df

# -----------------------------------------------------------------------------
# Indicator calculation & signal generation
# -----------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5 and MA30 columns to *df*."""
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma30"] = df["close"].rolling(window=30).mean()
    return df


def generate_signals(df: pd.DataFrame):
    """Create a *signal* column: 1 for golden cross, −1 for death cross."""
    df = df.dropna().copy()
    df["signal"] = 0

    # Golden cross
    cond_golden = (df["ma5"] > df["ma30"]) & (df["ma5"].shift() <= df["ma30"].shift())
    df.loc[cond_golden, "signal"] = 1

    # Death cross
    cond_death = (df["ma5"] < df["ma30"]) & (df["ma5"].shift() >= df["ma30"].shift())
    df.loc[cond_death, "signal"] = -1

    golden_dates = df.index[df["signal"] == 1]
    death_dates = df.index[df["signal"] == -1]
    return df, golden_dates, death_dates

# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------

def plot_analysis(df: pd.DataFrame, golden_dates, death_dates):
    """Produce the 4‑in‑1 diagnostic figure."""
    plt.figure(figsize=(12, 10))

    # Box plot
    plt.subplot(2, 2, 1)
    df["close"].plot.box()
    plt.title("Closing Price Box Plot")

    # Histogram
    plt.subplot(2, 2, 2)
    df["close"].plot.hist(bins=30)
    plt.title("Closing Price Histogram")

    # MA5 & MA30
    plt.subplot(2, 2, 3)
    df["ma5"].plot(label="MA5")
    df["ma30"].plot(label="MA30")
    plt.title("5‑Day and 30‑Day Moving Averages")
    plt.legend()

    # Price with crosses
    plt.subplot(2, 2, 4)
    df["close"].plot(label="Closing Price")
    plt.scatter(golden_dates, df.loc[golden_dates, "open"], marker="o", label="Golden Cross")
    plt.scatter(death_dates, df.loc[death_dates, "open"], marker="*", label="Death Cross")
    plt.title("Golden and Death Crosses")
    plt.legend()
    plt.grid(True)

    plt.show()

# -----------------------------------------------------------------------------
# Back‑test
# -----------------------------------------------------------------------------

def backtest(df: pd.DataFrame, initial_capital: float, start_date: str):
    """Simulate trading based on generated *signal* column and compute KPIs."""
    df = df[start_date:].dropna(subset=["open"]).copy()
    df, golden_dates, death_dates = generate_signals(df)

    cash = initial_capital
    shares = 0
    portfolio = [cash]

    for date in df.index:
        if pd.isna(df.loc[date, "open"]):
            continue
        if date in golden_dates:
            lots = cash // (100 * df.loc[date, "open"])
            shares += lots * 100
            cash -= lots * 100 * df.loc[date, "open"]
        elif date in death_dates and shares > 0:
            cash += shares * df.loc[date, "open"]
            shares = 0
        portfolio.append(cash + shares * df.loc[date, "open"])

    portfolio = pd.Series(portfolio).dropna()
    returns = portfolio.pct_change().dropna()
    cumulative = (1 + returns).cumprod()

    if cumulative.empty:
        print("Cumulative returns are empty. Please check the data.")
        return 0, 0, 0, 0

    total_ret = (portfolio.iloc[-1] - initial_capital) / initial_capital
    years = len(df) / 252.0
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

    # Plot cumulative
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative.index, cumulative, label="Cumulative Returns")
    plt.title("Cumulative Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    return total_ret, annual_ret, max_dd, sharpe

# -----------------------------------------------------------------------------
# Random‑Forest‑based cross date prediction
# -----------------------------------------------------------------------------

def _build_rf_dataset(df: pd.DataFrame, cross_dates: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix *X* and target vector *y* for RF training.

    *y* is the interval (in days) between consecutive crosses. Features are
    taken from the bar **before** the previous cross to avoid look‑ahead bias.
    """
    records = []
    targets = []
    crosses = cross_dates.sort_values()

    for i in range(1, len(crosses)):
        prev = crosses[i - 1]
        curr = crosses[i]
        interval = (curr - prev).days
        row = df.loc[prev]
        records.append({
            "ma_diff": row["ma5"] - row["ma30"],
            "ma_ratio": row["ma5"] / row["ma30"],
            "price": row["close"],
            "volatility20": df["close"].loc[:prev].pct_change().rolling(window=20).std().iloc[-1],
            "return_5": df["close"].pct_change(5).loc[prev]
        })
        targets.append(interval)

    X = pd.DataFrame(records).fillna(0)
    y = pd.Series(targets, name="interval")
    return X, y


def predict_next_cross_rf(df: pd.DataFrame):
    """Predict the next golden/death cross dates using a RandomForestRegressor."""

    # Build dataset from *all* crosses (golden + death) combined
    df, g_dates, d_dates = generate_signals(df)
    all_crosses = g_dates.union(d_dates).sort_values()

    if len(all_crosses) < 4:  # Need at least a few samples for RF
        print("Not enough crosses to train RandomForest; falling back to mean interval.")
        return _predict_next_cross_mean(df)

    X, y = _build_rf_dataset(df, all_crosses)

    # Train/test split just for MAE inspection (not strictly required for final prediction)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate quickly
    y_pred_test = model.predict(X_test)
    print(f"Random‑Forest MAE on hold‑out: {mean_absolute_error(y_test, y_pred_test):.2f} days")

    # Build latest feature vector (use last available bar)
    latest_row = df.iloc[-1]
    latest_features = pd.DataFrame({
        "ma_diff": [latest_row["ma5"] - latest_row["ma30"]],
        "ma_ratio": [latest_row["ma5"] / latest_row["ma30"]],
        "price": [latest_row["close"]],
        "volatility20": [df["close"].pct_change().rolling(window=20).std().iloc[-1]],
        "return_5": [df["close"].pct_change(5).iloc[-1]]
    }).fillna(0)

    predicted_interval = int(model.predict(latest_features)[0])
    last_cross_date = all_crosses[-1]
    next_cross_date = last_cross_date + pd.Timedelta(days=predicted_interval)

    # For demonstration we treat the *type* (golden or death) as alternating
    if last_cross_date in g_dates:
        next_golden = next_cross_date
        next_death = next_cross_date + pd.Timedelta(days=predicted_interval)
    else:
        next_death = next_cross_date
        next_golden = next_cross_date + pd.Timedelta(days=predicted_interval)

    return next_golden, next_death


# Fallback: simple mean interval (original behaviour)

def _predict_next_cross_mean(df: pd.DataFrame):
    df, g_dates, d_dates = generate_signals(df)
    all_crosses = g_dates.union(d_dates).sort_values()
    if len(all_crosses) < 2:
        return None, None
    intervals = np.diff(all_crosses).astype("timedelta64[D]").astype(int)
    avg_interval = int(np.mean(intervals))
    next_gold = df.index[-1] + pd.Timedelta(days=avg_interval)
    next_death = next_gold + pd.Timedelta(days=avg_interval)
    return next_gold, next_death

# -----------------------------------------------------------------------------
# Entry‑point CLI
# -----------------------------------------------------------------------------

def main():
    while True:
        print("---Stock Prediction and Recommendation System---")
        stock_code = input("Please enter stock code (e.g., AAPL for Apple): ")
        start_date = input("Please enter start date (YYYY-MM-DD): ")
        initial_capital = float(input("Please enter initial capital (positive number): "))

        df = get_stock_data(stock_code, start_date)
        df = calculate_indicators(df)
        df_signals, golden_dates, death_dates = generate_signals(df)

        plot_analysis(df_signals, golden_dates, death_dates)

        # Predict using RF; fallback to mean if needed
        next_golden, next_death = predict_next_cross_rf(df_signals)
        recommendation = "Buy" if next_golden and (next_golden < next_death) else "Hold"

        print(f"Latest Golden Cross: {golden_dates[-1] if len(golden_dates) > 0 else 'No data'}")
        print(f"Latest Death Cross: {death_dates[-1] if len(death_dates) > 0 else 'No data'}")
        print(f"Predicted Next Golden Cross: {next_golden if next_golden else 'No data'}")
        print(f"Predicted Next Death Cross: {next_death if next_death else 'No data'}")
        print(f"Recommendation: {recommendation}")

        total_ret, annual_ret, max_dd, sharpe = backtest(df_signals, initial_capital, start_date)
        print(f"Total Return: {total_ret:.2%}")
        print(f"Annualized Return: {annual_ret:.2%}")
        print(f"Maximum Drawdown: {max_dd:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")

        choice = input("Continue? (y/n): ")
        if choice.lower() != "y":
            break


if __name__ == "__main__":
    main()
