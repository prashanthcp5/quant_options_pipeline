import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, brier_score_loss
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MLPipeline")

class XGBoostRanker:
    """
    Trains and executes an XGBoost classifier to predict the probability 
    of an option contract hitting the profit target before the stop loss.
    """
    
    def __init__(self, model_path: str = "models/xgb_model.json"):
        self.model_path = Path(model_path)
        self.model = None
        
        # The exact features our FeatureEngineer outputs
        self.features = [
            'RSI_14', 'ATR_14', 'EMA_Alignment', 
            'Vol_OI_Ratio', 'Norm_Strike_Dist', 
            'Delta', 'Gamma', 'Theta', 'Vega', 'impliedVolatility'
        ]
        self.target = 'target_hit' # Binary: 1 (Won), 0 (Lost/Expired)

    def train(self, df: pd.DataFrame) -> None:
        """
        Trains the XGBoost model using Time-Series Cross-Validation 
        to ensure no look-ahead bias (data leakage).
        """
        logger.info(f"Starting model training with {len(df)} samples...")
        
        # Ensure data is sorted by date for time-series split
        df = df.sort_values(by='entry_date').reset_index(drop=True)
        
        X = df[self.features]
        y = df[self.target]

        # TimeSeriesSplit ensures we only train on the past to predict the future
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Basic XGBoost hyperparameters optimized for probability ranking
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100,
            'random_state': 42
        }

        self.model = xgb.XGBClassifier(**params)

        brier_scores = []
        precisions = []

        # Cross-validation loop
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            
            # Predict probabilities
            probs = self.model.predict_proba(X_test)[:, 1]
            preds = (probs > 0.5).astype(int)

            # Brier score measures accuracy of probabilistic predictions
            brier = brier_score_loss(y_test, probs)
            precision = precision_score(y_test, preds, zero_division=0)
            
            brier_scores.append(brier)
            precisions.append(precision)

        logger.info("Cross-Validation Complete.")
        logger.info(f"Average Brier Score: {np.mean(brier_scores):.4f}")
        logger.info(f"Average Precision: {np.mean(precisions):.4f}")

        # Retrain on the entire dataset for final model
        logger.info("Retraining on full dataset...")
        self.model.fit(X, y)
        self.save_model()

    def predict_signals(self, live_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes live options data, outputs confidence scores, and ranks them.
        """
        if self.model is None:
            self.load_model()
            
        X_live = live_features_df[self.features]
        
        # Get probability of class 1 (Hitting target)
        probabilities = self.model.predict_proba(X_live)[:, 1]
        
        live_features_df = live_features_df.copy()
        live_features_df['confidence_score'] = probabilities
        
        # Sort by highest confidence
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
        logger.info("Model loaded successfully.")

def _generate_mock_historical_data(num_samples: int = 1000) -> pd.DataFrame:
    """Helper to generate synthetic bootstrap data for initial run."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='h')
    
    df = pd.DataFrame({
        'entry_date': dates,
        'RSI_14': np.random.uniform(20, 80, num_samples),
        'ATR_14': np.random.uniform(1, 10, num_samples),
        'EMA_Alignment': np.random.choice([-1, 0, 1], num_samples),
        'Vol_OI_Ratio': np.random.uniform(0.1, 5.0, num_samples),
        'Norm_Strike_Dist': np.random.uniform(-3, 3, num_samples),
        'Delta': np.random.uniform(-1, 1, num_samples),
        'Gamma': np.random.uniform(0, 0.1, num_samples),
        'Theta': np.random.uniform(-0.5, 0, num_samples),
        'Vega': np.random.uniform(0, 1, num_samples),
        'impliedVolatility': np.random.uniform(0.1, 1.0, num_samples),
    })
    
    logit = (df['RSI_14'] - 50) * 0.05 + df['Vol_OI_Ratio'] * 0.5 + np.random.normal(0, 1, num_samples)
    probs = 1 / (1 + np.exp(-logit))
    df['target_hit'] = (probs > 0.5).astype(int)
    
    return df