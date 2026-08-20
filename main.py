import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from modules.storage_setup import DatabaseManager
from modules.data_ingestion import DataIngestionEngine
from modules.feature_engineering import FeatureEngineer
from modules.ml_pipeline import XGBoostRanker
from modules.signal_generator import SignalGenerator
from modules.forward_tester import ForwardTester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MainPipeline")

def run_pipeline(tickers: list[str]) -> None:
    logger.info("=== STARTING OPTIONS QUANT PIPELINE ===")

    logger.info("[Step 1/6] Initializing database storage...")
    db = DatabaseManager()
    db.initialize_schema()

    logger.info("[Step 2/6] Evaluating open trades against live market data...")
    tester = ForwardTester()
    tester.evaluate_open_signals()

    logger.info(f"[Step 3/6] Fetching and filtering option chains for: {tickers}")
    ingestion = DataIngestionEngine(tickers=tickers)
    raw_options = ingestion.get_filtered_options()

    if raw_options.empty:
        logger.warning("No options met the hard liquidity filters today. Exiting run.")
        tester.print_performance_metrics()
        return

    logger.info("[Step 4/6] Computing technical indicators and Greeks...")
    engineer = FeatureEngineer()
    featured_options = engineer.process_features(raw_options)

    if featured_options.empty:
        logger.warning("No valid options remaining after feature extraction. Exiting run.")
        tester.print_performance_metrics()
        return

    logger.info("[Step 5/6] Retraining ML model or applying Cold Start heuristic...")
    ranker = XGBoostRanker()
    historical_data = ranker.get_training_data()
    
    if len(historical_data) < 100:
        logger.warning("Cold Start: < 100 closed trades. Bypassing ML and applying Heuristic Baseline.")
        scored_options = featured_options.copy()
        
        # HEURISTIC SCORING: Low IV Rank + Higher Delta = Better Score 
        # This formula guarantees scores roughly scale between 0.55 and 0.85
        iv_factor = (100 - scored_options['IV_Rank']) / 100.0
        delta_factor = scored_options['Delta'].abs()
        
        scored_options['confidence_score'] = 0.55 + (0.15 * iv_factor) + (0.15 * delta_factor)
        scored_options['model_version'] = 'baseline_heuristic'
    else:
        ranker.train(historical_data)
        scored_options = ranker.predict_signals(featured_options)
        scored_options['model_version'] = 'xgb_v1'

    logger.info("[Step 6/6] Logging actionable trade signals to database...")
    generator = SignalGenerator()
    generator.generate_and_store_signals(scored_options)

    logger.info("=== PIPELINE EXECUTION COMPLETE ===")
    tester.print_performance_metrics()

if __name__ == "__main__":
    target_tickers = ["SPY", "QQQ", "AAPL", "NVDA", "SOFI", "F", "NIO"]
    run_pipeline(tickers=target_tickers)