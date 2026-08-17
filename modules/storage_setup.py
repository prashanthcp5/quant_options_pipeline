import sqlite3
import logging
from pathlib import Path

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("StorageSetup")

class DatabaseManager:
    """
    Handles connection management for the local SQLite database.
    """
    
    def __init__(self, db_path: str = "data/options_pipeline.db"):
        self.db_path = Path(db_path)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensures the parent directory for the local database exists."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created local directory: {self.db_path.parent}")

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured local sqlite3 connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row 
        return conn

    def initialize_schema(self) -> None:
        """
        Creates the necessary tables for signals and evaluations if they do not exist.
        """
        signals_table_sql = """
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,       
            underlying_ticker TEXT NOT NULL,
            option_symbol TEXT NOT NULL,
            option_type TEXT NOT NULL,        
            expiration_date TEXT NOT NULL,
            strike_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            entry_mark_price REAL NOT NULL,
            target_price REAL NOT NULL,       
            stop_loss_price REAL NOT NULL,    
            confidence_score REAL NOT NULL,   
            status TEXT DEFAULT 'OPEN',
            RSI_14 REAL,
            ATR_14 REAL,
            EMA_Alignment INTEGER,
            Vol_OI_Ratio REAL,
            Norm_Strike_Dist REAL,
            Delta REAL,
            Gamma REAL,
            Theta REAL,
            Vega REAL,
            impliedVolatility REAL,
            IV_Rank REAL
        );
        """

        evaluations_table_sql = """
        CREATE TABLE IF NOT EXISTS signal_evaluations (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT NOT NULL,
            check_timestamp TEXT NOT NULL,
            current_mark_price REAL NOT NULL,
            notes TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(signal_id)
        );
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(signals_table_sql)
                cursor.execute(evaluations_table_sql)
                conn.commit()
            logger.info(f"Database schema successfully verified locally at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

if __name__ == "__main__":
    db_manager = DatabaseManager()
    db_manager.initialize_schema()