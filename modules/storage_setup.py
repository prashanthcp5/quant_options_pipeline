import sqlite3
import logging
import os
import libsql_client
from pathlib import Path

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("StorageSetup")

class DatabaseManager:
    """
    Handles connection management for the database.
    Connects to Turso Cloud if credentials are present, otherwise falls back to local SQLite.
    """
    
    def __init__(self, db_path: str = "data/options_pipeline.db"):
        self.db_path = Path(db_path)
        self.turso_url = os.environ.get("TURSO_DATABASE_URL")
        self.turso_token = os.environ.get("TURSO_AUTH_TOKEN")
        
        # If running locally, ensure the local folder exists
        if not self.turso_url:
            self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensures the parent directory for the local database exists."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created local directory: {self.db_path.parent}")

    def get_connection(self):
        """
        Returns a configured database connection.
        If Turso env variables exist, returns a libsql client connection.
        If not, returns a standard local sqlite3 connection.
        """
        if self.turso_url and self.turso_token:
            # We are running in the cloud (GitHub Actions)
            try:
                # libsql connects to Turso using the same exact SQL syntax as SQLite
                conn = libsql_client.create_client_sync(
                    url=self.turso_url,
                    auth_token=self.turso_token
                )
                return conn
            except Exception as e:
                logger.error(f"Failed to connect to Turso Cloud Database: {e}")
                raise
        else:
            # We are running locally on your laptop
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
            # Check if we are using the local sqlite3 connection or the Turso libsql connection
            if self.turso_url and self.turso_token:
                conn = self.get_connection()
                conn.execute(signals_table_sql)
                conn.execute(evaluations_table_sql)
                logger.info("Database schema successfully verified on Turso Cloud.")
            else:
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
    # Test the initialization when the script is run directly
    db_manager = DatabaseManager()
    db_manager.initialize_schema()