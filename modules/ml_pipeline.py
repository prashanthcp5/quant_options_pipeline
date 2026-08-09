import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, brier_score_loss
import logging
import json
from pathlib import Path
from modules.storage_setup import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MLPipeline")

class XGBoostRanker:
    def __init__(self, model_path: str = "models/xgb_model.json"):
        self.model_path = Path(model_path)
        self.model = None
        self.db = DatabaseManager()
        
        # PRUNED FEATURE LIST - Focusing only on the top 5 heavyweights
        self.features = [
            'Delta', 'RSI_14', 'Norm_Strike_Dist', 
            'ATR_14', 'impliedVolatility'
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
            
            # Drop rows where any of our 5 features are missing
            df = df.dropna(subset=self.features)
        return df

    def train(self, df: pd.DataFrame) -> None:
        logger.info(f"Training pruned model on {len(df)} real historical trades...")
        X = df[self.features]
        y = df[self.target]

        tscv = TimeSeriesSplit(n_splits=5)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'learning_rate': 0.01,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'n_estimators': 50,
            'random_state': 42
        }

        self.model = xgb.XGBClassifier(**params)

        brier_scores = []
        precisions = []

        if len(df) > 50:
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                self.model.fit(X_train, y_train)
                
                probs = self.model.predict_proba(X_test)[:, 1]
                preds = (probs > 0.5).astype(int)

                brier = brier_score_loss(y_test, probs)
                precision = precision_score(y_test, preds, zero_division=0)
                
                brier_scores.append(brier)
                precisions.append(precision)

            logger.info("Cross-Validation Complete.")
            logger.info(f"Average Brier Score: {np.mean(brier_scores):.4f}")
            logger.info(f"Average Precision: {np.mean(precisions):.4f}")

        # Final retrain on the entire dataset
        self.model.fit(X, y)
        self.save_model()
        self.print_feature_importance()

    def print_feature_importance(self) -> None:
        if self.model is None:
            return
        
        importance = self.model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': self.features, 'Importance': importance})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
        
        print("\n" + "="*40)
        print("PRUNED XGBOOST FEATURE IMPORTANCE")
        print("="*40)
        for _, row in feat_imp.iterrows():
            print(f"{row['Feature']:<20}: {row['Importance']:.4f}")
        print("="*40 + "\n")

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