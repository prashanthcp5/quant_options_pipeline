import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import logging
from modules.ml_pipeline import XGBoostRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HyperTuner")

def run_hyperparameter_tuning():
    logger.info("Initializing Hyperparameter Tuner...")
    ranker = XGBoostRanker()
    df = ranker.get_training_data()
    
    if len(df) < 100:
        logger.error("Not enough data to tune.")
        return
        
    X = df[ranker.features]
    y = df[ranker.target]
    
    # TimeSeries split to prevent look-ahead bias during testing
    tscv = TimeSeriesSplit(n_splits=5)
    
    # The grid of parameters we want the computer to test
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.7, 0.8, 1.0]
    }
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    logger.info("Starting Grid Search. This will train 300+ models. Please wait...")
    
    # n_jobs=-1 tells your laptop to use every CPU core it has to speed this up
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='precision',
        cv=tscv,
        verbose=1,
        n_jobs=-1 
    )
    
    grid_search.fit(X, y)
    
    print("\n" + "="*40)
    print("OPTIMAL HYPERPARAMETERS FOUND")
    print("="*40)
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best Cross-Validated Precision: {grid_search.best_score_:.4f}")
    print("="*40 + "\n")
    
    print("Next step: Update the params dictionary in modules/ml_pipeline.py with these exact values!")

if __name__ == "__main__":
    run_hyperparameter_tuning()