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
    Handles the initialization and connection management for the local SQLite database.
    """
    
    def __init__(self, db_path: str = "data/options_pipeline.db"):
        self.db_path = Path(db_path)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensures the parent directory for the database exists."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.db_path.parent}")

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        # Allows accessing columns by name (e.g., row['symbol'])
        conn.row_factory = sqlite3.Row 
        return conn

    def initialize_schema(self) -> None:
        """
        Creates the necessary tables for signals and evaluations if they do not exist.
        """
        signals_table_sql = """
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,       -- e.g., UUID or OptionSymbol_Date
            underlying_ticker TEXT NOT NULL,
            option_symbol TEXT NOT NULL,
            option_type TEXT NOT NULL,        -- 'CALL' or 'PUT'
            expiration_date TEXT NOT NULL,
            strike_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            entry_mark_price REAL NOT NULL,
            target_price REAL NOT NULL,       -- +50% target
            stop_loss_price REAL NOT NULL,    -- -30% stop
            confidence_score REAL NOT NULL,   -- ML probability output
            status TEXT DEFAULT 'OPEN'        -- 'OPEN', 'WON', 'LOST', 'EXPIRED'
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
            logger.info(f"Database schema successfully initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

if __name__ == "__main__":
    # Test the initialization when the script is run directly
    db_manager = DatabaseManager()
    db_manager.initialize_schema()