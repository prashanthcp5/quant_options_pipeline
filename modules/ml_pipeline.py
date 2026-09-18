import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, roc_auc_score
import logging
from pathlib import Path
from modules.storage_setup import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MLPipeline")

class XGBoostRanker:
    # Capacity is scaled to how much training data actually exists. Early on
    # (a few hundred rows, ~50 wins) a deep model memorized the data - train
    # logloss collapsed to 0.003 and it assigned 96% confidence to trades that
    # objectively lost. With thousands of rows and hundreds of wins, that same
    # capacity becomes useful instead of harmful. These thresholds let the
    # model grow with the dataset rather than being re-tuned by hand.
    CAPACITY_TIERS = [
        # (min_rows, min_positives, params, label)
        (1500, 300, dict(max_depth=5, learning_rate=0.05, n_estimators=200), "HIGH"),
        (600,  120, dict(max_depth=4, learning_rate=0.03, n_estimators=100), "MEDIUM"),
        (0,      0, dict(max_depth=3, learning_rate=0.01, n_estimators=50),  "LOW"),
    ]

    def __init__(self, model_path: str = "models/xgb_model.json"):
        self.model_path = Path(model_path)
        self.model = None
        self.db = DatabaseManager()

        self.features = [
            'Delta', 'RSI_14', 'Norm_Strike_Dist',
            'ATR_14', 'IV_Rank'
        ]
        self.target = 'target_hit'

    def get_training_data(self) -> pd.DataFrame:
        """Pulls all closed trades that have recorded feature data."""
        query = "SELECT * FROM signals WHERE status IN ('WON', 'LOST') AND RSI_14 IS NOT NULL"
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        if not df.empty:
            df['target_hit'] = (df['status'] == 'WON').astype(int)
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df = df.sort_values('entry_date').reset_index(drop=True)

            if 'IV_Rank' not in df.columns and 'impliedVolatility' in df.columns:
                logger.warning("IV_Rank missing from old DB schema. Temporarily mapping impliedVolatility to IV_Rank for backward compatibility.")
                df['IV_Rank'] = df['impliedVolatility'] * 100

            df = df.dropna(subset=self.features)
        return df

    def _select_capacity(self, n_rows: int, n_positives: int) -> tuple:
        for min_rows, min_pos, params, label in self.CAPACITY_TIERS:
            if n_rows >= min_rows and n_positives >= min_pos:
                return params, label
        return self.CAPACITY_TIERS[-1][2], self.CAPACITY_TIERS[-1][3]

    def _goldilocks_precision(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        """Precision among trades scoring in the deployed 0.60-0.80 band."""
        mask = (probs >= 0.60) & (probs <= 0.80)
        if not np.any(mask):
            return 0.0
        y_true_filtered = y_true[mask]
        return np.sum(y_true_filtered) / len(y_true_filtered)

    def train(self, df: pd.DataFrame) -> None:
        logger.info(f"Training model on {len(df)} real historical trades...")
        X = df[self.features]
        y = df[self.target]

        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        dynamic_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        logger.info(f"Class Imbalance -> Won: {pos_count}, Lost: {neg_count}. Applying scale_pos_weight: {dynamic_weight:.2f}")

        capacity_params, capacity_label = self._select_capacity(len(df), pos_count)
        logger.info(
            f"Capacity tier: {capacity_label} "
            f"(depth={capacity_params['max_depth']}, lr={capacity_params['learning_rate']}, "
            f"n_estimators={capacity_params['n_estimators']})"
        )

        tscv = TimeSeriesSplit(n_splits=5)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'scale_pos_weight': dynamic_weight,
            'random_state': 42,
            **capacity_params,
        }

        self.model = xgb.XGBClassifier(**params)

        brier_scores, goldilocks_precisions, auc_scores = [], [], []
        train_losses, test_losses = [], []

        if len(df) > 50:
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                self.model.fit(X_train, y_train)
                probs = self.model.predict_proba(X_test)[:, 1]
                train_probs = self.model.predict_proba(X_train)[:, 1]

                brier_scores.append(brier_score_loss(y_test, probs))
                goldilocks_precisions.append(self._goldilocks_precision(y_test.values, probs))
                train_losses.append(brier_score_loss(y_train, train_probs))
                test_losses.append(brier_score_loss(y_test, probs))

                if len(np.unique(y_test)) > 1:
                    auc_scores.append(roc_auc_score(y_test, probs))

            logger.info("Cross-Validation Complete.")
            logger.info(f"Average Brier Score: {np.mean(brier_scores):.4f}")
            logger.info(f"Goldilocks Zone Precision: {np.mean(goldilocks_precisions):.4f}")
            if auc_scores:
                logger.info(f"ROC AUC Score: {np.mean(auc_scores):.4f}")

            # Overfitting tripwire. A large train/test Brier gap is exactly the
            # signature that flagged the earlier memorization problem, so it is
            # surfaced explicitly rather than left to be discovered from live
            # results days later.
            gap = np.mean(test_losses) - np.mean(train_losses)
            logger.info(f"Train->Test Brier gap: {gap:+.4f}")
            if gap > 0.08:
                logger.warning(
                    f"OVERFITTING WARNING: train/test Brier gap of {gap:+.4f} is large. "
                    f"The {capacity_label} capacity tier may be too high for {len(df)} rows / "
                    f"{pos_count} wins. Treat this model's confidence scores with suspicion."
                )

        self.model.fit(X, y)
        self.save_model()
        self.print_feature_importance()

    def print_feature_importance(self) -> None:
        if self.model is None:
            return
        importance = self.model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': self.features, 'Importance': importance})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
        print("\n" + "=" * 40)
        print("XGBOOST FEATURE IMPORTANCE")
        print("=" * 40)
        for _, row in feat_imp.iterrows():
            print(f"{row['Feature']:<20}: {row['Importance']:.4f}")
        print("=" * 40 + "\n")

    def predict_signals(self, live_features_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self.load_model()
        X_live = live_features_df[self.features]
        probabilities = self.model.predict_proba(X_live)[:, 1]
        live_features_df = live_features_df.copy()
        live_features_df['confidence_score'] = probabilities
        live_features_df = live_features_df.sort_values(by='confidence_score', ascending=False)
        return live_features_df

    def save_model(self) -> None:
        if not self.model_path.parent.exists():
            self.model_path.parent.mkdir(parents=True)
        self.model.save_model(self.model_path)
        logger.info(f"Model saved successfully to {self.model_path}")

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"No trained model found at {self.model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)