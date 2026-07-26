import pandas as pd
import uuid
from datetime import datetime
import logging
from modules.storage_setup import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SignalGenerator")

class SignalGenerator:
    """
    Takes ML-scored options, filters for high confidence, calculates 
    risk-management targets, and stores the trades in the database.
    """
    
    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold
        self.db = DatabaseManager()
        
        # Risk Management Rules
        self.profit_target_pct = 0.50  # +50%
        self.stop_loss_pct = 0.30      # -30%

    def generate_and_store_signals(self, scored_options_df: pd.DataFrame) -> None:
        """
        Filters options above the confidence threshold and logs them into SQLite.
        """
        if scored_options_df.empty or 'confidence_score' not in scored_options_df.columns:
            logger.warning("No scored options provided to generate signals.")
            return

        # Filter for high-probability setups
        actionable_signals = scored_options_df[
            scored_options_df['confidence_score'] >= self.confidence_threshold
        ].copy()

        if actionable_signals.empty:
            logger.info(f"No signals met the confidence threshold of {self.confidence_threshold}.")
            return

        logger.info(f"Found {len(actionable_signals)} actionable signals. Storing to database...")
        
        today_str = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        inserted_count = 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in actionable_signals.iterrows():
                # Generate a unique ID for this specific trade signal
                signal_id = f"SIG_{uuid.uuid4().hex[:8].upper()}"
                
                # Calculate targets based on the current option mark price
                entry_price = row.get('mark_price', 0.0)
                
                # Skip if we have invalid pricing
                if entry_price <= 0:
                    continue
                    
                target_price = entry_price * (1 + self.profit_target_pct)
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)

                # Prepare the SQL insert
                insert_sql = """
                INSERT INTO signals (
                    signal_id, underlying_ticker, option_symbol, option_type, 
                    expiration_date, strike_price, entry_date, entry_mark_price, 
                    target_price, stop_loss_price, confidence_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                params = (
                    signal_id,
                    row['underlying_ticker'],
                    row['contractSymbol'],
                    row['option_type'],
                    row['expiration_date'],
                    row['strike'],
                    today_str,
                    round(entry_price, 2),
                    round(target_price, 2),
                    round(stop_loss_price, 2),
                    round(row['confidence_score'], 4),
                    'OPEN'
                )
                
                try:
                    cursor.execute(insert_sql, params)
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Failed to insert signal {signal_id}: {e}")
            
            conn.commit()
            
        logger.info(f"Successfully saved {inserted_count} new signals to the database.")