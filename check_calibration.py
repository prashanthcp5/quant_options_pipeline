"""
Standalone model-health check for quant_options_pipeline.

Run this anytime against your local data/options_pipeline.db to see whether
confidence_score is actually predictive of outcome yet. No dependency on the
rest of the pipeline modules - just point it at the DB file.

Usage:
    python check_calibration.py
    python check_calibration.py --db path/to/options_pipeline.db
    python check_calibration.py --min-trades 30
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_closed_trades(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM signals WHERE status IN ('WON', 'LOST')", conn
    )
    conn.close()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["y"] = (df["status"] == "WON").astype(int)
    return df.sort_values("entry_date").reset_index(drop=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/options_pipeline.db")
    parser.add_argument(
        "--min-trades",
        type=int,
        default=30,
        help="Minimum closed trades before metrics are considered meaningful.",
    )
    args = parser.parse_args()

    df = load_closed_trades(args.db)

    if df.empty:
        print("No closed (WON/LOST) trades yet. Nothing to check.")
        sys.exit(0)

    print_header("SAMPLE SIZE")
    print(f"Total closed trades: {len(df)}")
    print(f"Date range: {df['entry_date'].min().date()} -> {df['entry_date'].max().date()}")
    print(f"Unique trading days: {df['entry_date'].dt.date.nunique()}")
    if len(df) < args.min_trades:
        print(
            f"\n[!] Fewer than {args.min_trades} closed trades. "
            "Treat everything below as noisy / directional only."
        )

    print_header("OVERALL WIN RATE")
    win_rate = df["y"].mean()
    print(f"Win rate: {win_rate:.2%}  ({df['y'].sum()} WON / {len(df)} total)")
    print("Breakeven win rate for +50%/-30% target/stop: 37.50%")

    print_header("IS CONFIDENCE SCORE PREDICTIVE?")
    corr = np.corrcoef(df["confidence_score"], df["y"])[0, 1]
    print(f"Point-biserial correlation (confidence vs outcome): {corr:+.3f}")
    print("  Positive = good (higher confidence -> more wins)")
    print("  Negative = bad  (higher confidence -> more losses, still inverted)")

    if df["y"].nunique() > 1:
        auc = roc_auc_score(df["y"], df["confidence_score"])
        print(f"\nROC AUC (confidence_score as ranking signal): {auc:.3f}")
        print("  0.50 = random  |  >0.55 = weak signal  |  >0.65 = decent signal  |  <0.45 = inverted")
    else:
        print("\nROC AUC: N/A (only one outcome class present so far)")

    print_header("CALIBRATION BY CONFIDENCE BUCKET")
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    df["bucket"] = pd.cut(df["confidence_score"], bins=bins, labels=labels, include_lowest=True)
    bucket_stats = df.groupby("bucket", observed=True).agg(
        n=("y", "size"), win_rate=("y", "mean")
    )
    bucket_stats["win_rate"] = (bucket_stats["win_rate"] * 100).round(1)
    print(bucket_stats.to_string())
    print(
        "\nHealthy pattern: win_rate should generally RISE as the bucket rises. "
        "If it's flat or falling, confidence still isn't tracking reality."
    )

    print_header("GOLDILOCKS ZONE (0.60-0.80) SPECIFIC CHECK")
    zone = df[(df["confidence_score"] >= 0.60) & (df["confidence_score"] <= 0.80)]
    if zone.empty:
        print("No closed trades fall in the 0.60-0.80 zone yet.")
    else:
        zone_win_rate = zone["y"].mean()
        print(f"Trades in zone: {len(zone)}")
        print(f"Win rate in zone: {zone_win_rate:.2%}")
        print(f"Need >= 37.50% just to break even on the +50%/-30% target/stop.")

    print_header("PER-TICKER BREAKDOWN")
    if "underlying_ticker" in df.columns:
        ticker_stats = df.groupby("underlying_ticker").agg(
            n=("y", "size"), win_rate=("y", "mean")
        )
        ticker_stats["win_rate"] = (ticker_stats["win_rate"] * 100).round(1)
        print(ticker_stats.sort_values("n", ascending=False).to_string())

    print("\nDone.\n")


if __name__ == "__main__":
    main()