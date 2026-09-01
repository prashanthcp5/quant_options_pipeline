import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from sklearn.metrics import log_loss, brier_score_loss

# Suppress XGBoost warnings for clean output
logging.getLogger("xgboost").setLevel(logging.ERROR)


def run_comparison():
    print("\nLoading historical data from options_pipeline.db...")
    conn = sqlite3.connect('data/options_pipeline.db')

    query = "SELECT * FROM signals WHERE status IN ('WON', 'LOST') AND RSI_14 IS NOT NULL"
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("No closed trades found.")
        return

    # Sort chronologically BEFORE splitting - a random split would leak future
    # trades into the training set and hide overfitting just as badly as not
    # splitting at all.
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df = df.sort_values('entry_date').reset_index(drop=True)

    y = (df['status'] == 'WON').astype(int)

    features = ['Delta', 'RSI_14', 'Norm_Strike_Dist', 'ATR_14']
    if 'IV_Rank' in df.columns:
        features.append('IV_Rank')
    else:
        features.append('impliedVolatility')

    df = df.dropna(subset=features)
    X = df[features]
    y = y.loc[df.index]

    # Chronological 80/20 holdout. With <500 rows this is a rough cut, not a
    # rigorous validation scheme - treat it as a smoke test for "does this
    # config generalize at all", not a final answer.
    split_idx = int(len(df) * 0.8)
    X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    dynamic_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    print(f"Full dataset: {len(df)} trades ({y.sum()} WON / {len(y) - y.sum()} LOST)")
    print(f"Train split:  {len(X_train)} trades ({pos_count} WON / {neg_count} LOST)")
    print(f"Holdout split: {len(X_holdout)} trades ({y_holdout.sum()} WON / {len(y_holdout) - y_holdout.sum()} LOST)")
    print(f"Calculated scale_pos_weight (from train split): {dynamic_weight:.2f}\n")

    if y_holdout.nunique() < 2:
        print("[!] Holdout set has only one outcome class - Brier/logloss on holdout")
        print("    will be uninformative. Treat holdout numbers below with caution")
        print("    until more WON examples accumulate.\n")

    configs = {
        "OLD (no weighting, weak hyperparams)": {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'learning_rate': 0.01,
            'n_estimators': 50,
            'random_state': 42,
        },
        "FRIEND'S NEW (weighted, upgraded hyperparams)": {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'scale_pos_weight': dynamic_weight,
            'random_state': 42,
        },
        "CLAUDE'S CONSERVATIVE (weighted, cautious hyperparams)": {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'learning_rate': 0.01,
            'n_estimators': 50,
            'scale_pos_weight': dynamic_weight,
            'random_state': 42,
        },
    }

    results = []

    for name, params in configs.items():
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        probs_train = model.predict_proba(X_train)[:, 1]
        probs_holdout = model.predict_proba(X_holdout)[:, 1]

        train_loss = log_loss(y_train, probs_train, labels=[0, 1])
        train_brier = brier_score_loss(y_train, probs_train)

        if y_holdout.nunique() == 2:
            holdout_loss = log_loss(y_holdout, probs_holdout, labels=[0, 1])
            holdout_brier = brier_score_loss(y_holdout, probs_holdout)
        else:
            holdout_loss = float('nan')
            holdout_brier = float('nan')

        gold_train = np.sum((probs_train >= 0.60) & (probs_train <= 0.80))
        gold_holdout = np.sum((probs_holdout >= 0.60) & (probs_holdout <= 0.80))

        print(f"--- {name} ---")
        print(f"  TRAIN   -> logloss: {train_loss:.4f} | brier: {train_brier:.4f} | "
              f"mean: {probs_train.mean():.3f} | max: {probs_train.max():.3f} | "
              f"in zone: {gold_train}/{len(probs_train)}")
        print(f"  HOLDOUT -> logloss: {holdout_loss:.4f} | brier: {holdout_brier:.4f} | "
              f"mean: {probs_holdout.mean():.3f} | max: {probs_holdout.max():.3f} | "
              f"in zone: {gold_holdout}/{len(probs_holdout)}")

        gap = holdout_loss - train_loss if not np.isnan(holdout_loss) else float('nan')
        if not np.isnan(gap):
            flag = "  [!] Large train/holdout gap - possible overfitting" if gap > 0.3 else ""
            print(f"  Train->Holdout logloss gap: {gap:+.4f}{flag}")
        print("-" * 60 + "\n")

        results.append({
            "config": name,
            "train_logloss": train_loss,
            "holdout_logloss": holdout_loss,
            "holdout_in_zone": gold_holdout,
        })

    print("Summary: prefer the config with the LOWEST holdout logloss/brier and a")
    print("small train->holdout gap, not the one with the most trades in the zone")
    print("on training data alone - that number is exactly what overfitting inflates.")


if __name__ == "__main__":
    run_comparison()