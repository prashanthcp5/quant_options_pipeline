"""
Phase 1: Signal Validation (no options data required)

Tests whether the pipeline's core technical features have any real predictive
power over forward stock returns, using free daily OHLCV data across a broad,
diversified, affordable universe. This is deliberately decoupled from options
pricing entirely - if there's no signal here, no amount of options-structure
engineering downstream will manufacture one.

Method: for each trading day, compute the cross-sectional Spearman rank
correlation ("Information Coefficient", IC) between each feature and the
forward N-day return across the whole universe. A real, reusable signal shows
up as a consistently non-zero mean IC over time, not just a lucky stretch.

This does NOT train a model or make any trading decision - it's a pure
diagnostic to decide whether Phase 2 (options backtesting) is worth doing.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

print(">>> Phase 1 script starting - if you don't see progress lines below within a few seconds, something is wrong.", flush=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger("Phase1SignalValidation")

# Broad, diversified candidate universe spanning multiple sectors - deliberately
# NOT just index-correlated mega-caps, to avoid the SPY/QQQ correlated-loss
# problem the live pipeline already ran into. The script filters this down to
# your actual price band at runtime, so it stays correct even as prices move.
CANDIDATE_UNIVERSE = [
    # Financials
    "BAC", "WFC", "C", "SOFI", "PYPL", "COF", "ALLY", "SCHW", "KEY", "RF",
    # Tech / semis / software
    "INTC", "CSCO", "HPQ", "DELL", "AMD", "PLTR", "SNAP", "UBER", "LYFT", "PINS", "DOCU",
    # Consumer discretionary
    "F", "GM", "NIO", "RIVN", "LCID", "SBUX", "TGT", "NKE", "GAP", "M",
    # Consumer staples
    "KO", "MO", "KHC", "GIS", "KMB", "CAG",
    # Industrials / airlines
    "CSX", "DAL", "UAL", "CCL", "NCLH", "LUV",
    # Healthcare
    "PFE", "MRK", "BMY", "GILD", "VTRS", "CVS",
    # Energy
    "OXY", "MRO", "DVN", "SLB", "HAL",
    # Communication
    "T", "VZ", "CMCSA", "WBD", "PARA",
    # Materials
    "FCX", "NUE", "X", "CLF",
    # Utilities
    "AES", "NRG",
]

PRICE_BAND = (10.0, 150.0)   # matches a $300-500 account buying affordable contracts
LOOKBACK_YEARS = 5
FORWARD_DAYS = 10            # forward return horizon; roughly matches a short-dated option hold
MIN_HISTORY_ROWS = 300       # skip tickers with too little history (recent IPOs, etc.)


def filter_universe_by_price(tickers: list, price_band: tuple) -> list:
    """Keep only tickers currently trading within the target affordability band."""
    print(f">>> Checking current price for {len(tickers)} candidate tickers...", flush=True)
    keep = []
    for idx, t in enumerate(tickers, 1):
        print(f"  [{idx}/{len(tickers)}] {t}...", end=" ", flush=True)
        try:
            hist = yf.Ticker(t).history(period="5d")
            if hist.empty:
                continue
            last_price = hist["Close"].iloc[-1]
            if price_band[0] <= last_price <= price_band[1]:
                keep.append(t)
                print(f"OK (${last_price:.2f})", flush=True)
            else:
                print(f"skip (${last_price:.2f}, outside band)", flush=True)
        except Exception as e:
            print(f"FAILED ({e})", flush=True)
    return keep


def fetch_history(ticker: str, years: int) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    try:
        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df.empty or len(df) < MIN_HISTORY_ROWS:
            return None
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch history for {ticker}: {e}")
        return None


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_realized_vol_rank(close: pd.Series, vol_window: int = 30, rank_window: int = 252) -> pd.Series:
    """
    Stand-in for IV_Rank (no historical IV data available for free): ranks
    current 30-day realized volatility against its own trailing 1-year range.
    This is an approximation, not a substitute for real implied vol - flagged
    clearly so it isn't mistaken for the live pipeline's IV_Rank later.
    """
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    roll_min = realized_vol.rolling(rank_window).min()
    roll_max = realized_vol.rolling(rank_window).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return ((realized_vol - roll_min) / denom * 100).clip(0, 100)


def compute_features_and_forward_return(df: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["RSI_14"] = compute_rsi(df["Close"])
    # ATR_14_pct, not raw ATR_14: raw dollar ATR mechanically scales with price
    # level (a $130 stock has a bigger dollar ATR than a $13 stock even at
    # identical relative volatility), which would make the cross-sectional
    # ranking mostly reflect "is this a higher-priced stock" rather than any
    # real volatility signal. Normalizing by price fixes this.
    out["ATR_14_pct"] = compute_atr(df) / df["Close"]
    out["RealizedVolRank"] = compute_realized_vol_rank(df["Close"])
    out["forward_return"] = df["Close"].shift(-forward_days) / df["Close"] - 1
    return out.dropna()


def compute_daily_ic(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    For each date, computes the cross-sectional Spearman rank correlation
    between each feature and the forward return, across all tickers that
    have data that day. This is the standard first diagnostic a quant runs
    before ever fitting a model - it isolates "does this feature rank stocks
    correctly" from any model-fitting noise.
    """
    records = []
    for date, group in panel.groupby("date"):
        if len(group) < 10:  # need enough cross-sectional breadth that day
            continue
        row = {"date": date, "n_stocks": len(group)}
        for feat in feature_cols:
            if group[feat].nunique() > 1:
                ic, _ = stats.spearmanr(group[feat], group["forward_return"])
                row[feat] = ic
            else:
                row[feat] = np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("date")


def summarize_ic(ic_series: pd.Series, feature_name: str, n_windows: int = 4) -> None:
    ic_series = ic_series.dropna()
    if len(ic_series) < 30:
        print(f"{feature_name}: insufficient IC observations ({len(ic_series)}) to summarize.")
        return

    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat, p_value = stats.ttest_1samp(ic_series, 0)

    print(f"\n--- {feature_name} ---")
    print(f"  Mean IC: {mean_ic:+.4f}   IC IR: {ic_ir:+.3f}   t-stat: {t_stat:+.2f}   p-value: {p_value:.4f}")
    print(f"  ({'statistically distinguishable from zero' if p_value < 0.05 else 'NOT statistically distinguishable from zero'} at 5% level)")

    # Split into non-overlapping windows - a real, reusable signal should show
    # up in most windows, not just one lucky stretch (the exact failure mode
    # that kept fooling the live dashboard on small samples).
    window_size = len(ic_series) // n_windows
    print(f"  Robustness across {n_windows} non-overlapping time windows:")
    for i in range(n_windows):
        window = ic_series.iloc[i * window_size:(i + 1) * window_size]
        if len(window) > 5:
            print(f"    Window {i+1} ({window.index.min().date()} to {window.index.max().date()}): mean IC = {window.mean():+.4f}")


def main():
    logger.info(f"Filtering {len(CANDIDATE_UNIVERSE)} candidates to price band {PRICE_BAND}...")
    universe = filter_universe_by_price(CANDIDATE_UNIVERSE, PRICE_BAND)
    logger.info(f"{len(universe)}/{len(CANDIDATE_UNIVERSE)} candidates in band: {universe}")

    if len(universe) < 15:
        logger.warning("Fewer than 15 tickers survived the price filter - consider widening PRICE_BAND.")

    feature_cols = ["RSI_14", "ATR_14_pct", "RealizedVolRank"]
    all_panels = []

    print(f"\n>>> Fetching {LOOKBACK_YEARS}y history for {len(universe)} tickers...", flush=True)
    for idx, ticker in enumerate(universe, 1):
        print(f"  [{idx}/{len(universe)}] {ticker}...", end=" ", flush=True)
        raw = fetch_history(ticker, LOOKBACK_YEARS)
        if raw is None:
            print("no usable history, skipping", flush=True)
            continue
        print(f"{len(raw)} rows OK", flush=True)
        feats = compute_features_and_forward_return(raw, FORWARD_DAYS)
        if feats.empty:
            continue
        feats["ticker"] = ticker
        feats["date"] = feats.index
        all_panels.append(feats)

    if not all_panels:
        logger.error("No usable data collected. Check network access and ticker list.")
        return

    panel = pd.concat(all_panels, ignore_index=True)
    logger.info(f"Combined panel: {len(panel)} stock-days across {panel['ticker'].nunique()} tickers.")

    ic_df = compute_daily_ic(panel, feature_cols)
    ic_df.to_csv("research/phase1_ic_results.csv")
    logger.info("Saved daily IC series to research/phase1_ic_results.csv")

    print("\n" + "=" * 60)
    print(f"PHASE 1 RESULTS: {feature_cols} vs {FORWARD_DAYS}-day forward return")
    print("=" * 60)
    for feat in feature_cols:
        summarize_ic(ic_df[feat], feat)
    print("\n" + "=" * 60)
    print("A feature worth carrying into Phase 2 should show a mean IC with")
    print("p < 0.05 AND a consistent sign across most/all time windows above.")
    print("A feature that's only significant in one window is noise, not signal -")
    print("this is the same 'looked good on one slice of data' trap that fooled")
    print("the live dashboard's early readings on small samples.")
    print("=" * 60)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        print("\n>>> SCRIPT CRASHED - full error below:", flush=True)
        traceback.print_exc()
        input("\nPress Enter to close...")  # keeps the window open so you can read the error