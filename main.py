import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path for relative module imports
sys.path.append(str(Path(__file__).resolve().parent))

from modules.storage_setup import DatabaseManager
from modules.data_ingestion import DataIngestionEngine
from modules.feature_engineering import FeatureEngineer
from modules.ml_pipeline import XGBoostRanker, _generate_mock_historical_data
from modules.signal_generator import SignalGenerator
from modules.forward_tester import ForwardTester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MainPipeline")

def run_pipeline(tickers: list[str]) -> None:
    """
    Executes the full options scanning, ML ranking, and forward testing pipeline.
    """
    logger.info("=== STARTING OPTIONS QUANT PIPELINE ===")

    # 1. Initialize Storage
    logger.info("[Step 1/6] Initializing database storage...")
    db = DatabaseManager()
    db.initialize_schema()

    # 2. Forward Test existing open signals
    logger.info("[Step 2/6] Evaluating open trades against live market data...")
    tester = ForwardTester()
    tester.evaluate_open_signals()

    # 3. Data Ingestion & Pre-Filtering
    logger.info(f"[Step 3/6] Fetching and filtering option chains for: {tickers}")
    ingestion = DataIngestionEngine(tickers=tickers)
    raw_options = ingestion.get_filtered_options()

    if raw_options.empty:
        logger.warning("No options met the hard liquidity/DTE filters today. Exiting run.")
        tester.print_performance_metrics()
        return

    # 4. Feature Engineering
    logger.info("[Step 4/6] Computing technical indicators and Black-Scholes Greeks...")
    engineer = FeatureEngineer()
    featured_options = engineer.process_features(raw_options)

    if featured_options.empty:
        logger.warning("No valid options remaining after feature extraction. Exiting run.")
        tester.print_performance_metrics()
        return

    # 5. ML Scoring & Ranking
    logger.info("[Step 5/6] Scoring options with XGBoost Ranker...")
    ranker = XGBoostRanker()
    
    # Check if model exists; if not, train a starter model on bootstrap data
    if not Path("models/xgb_model.json").exists():
        logger.info("No trained model found. Training initial baseline model...")
        mock_data = _generate_mock_historical_data(1500)
        ranker.train(mock_data)

    scored_options = ranker.predict_signals(featured_options)

    # 6. Signal Generation & Database Storage
    logger.info("[Step 6/6] Logging actionable trade signals to database...")
    # Setting threshold to 0.50 so we catch actionable setups for testing
    generator = SignalGenerator(confidence_threshold=0.50)
    generator.generate_and_store_signals(scored_options)

    # Final Summary
    logger.info("=== PIPELINE EXECUTION COMPLETE ===")
    tester.print_performance_metrics()

if __name__ == "__main__":
    # Highly liquid tickers to scan
    target_tickers = ["SPY", "QQQ", "AAPL", "NVDA"]
    run_pipeline(tickers=target_tickers)