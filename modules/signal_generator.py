import pandas as pd
import uuid
from datetime import datetime
import logging
from modules.storage_setup import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SignalGenerator")

class SignalGenerator:
    # Removed min/max confidence parameters; the generator now logs the entire market
    def __init__(self):
        self.db = DatabaseManager()
        self.profit_target_pct = 0.50  
        self.stop_loss_pct = 0.30      

    def generate_and_store_signals(self, scored_options_df: pd.DataFrame) -> None:
        if scored_options_df.empty or 'confidence_score' not in scored_options_df.columns:
            return

        # Cap exposure per ticker, but rank by confidence across BOTH directions
        # combined, rather than per (ticker, option_type). The old grouping forced
        # 3 calls AND 3 puts on every ticker regardless of the model's actual
        # conviction, which meant deliberately taking the "other side" of the
        # model's own view just to fill a quota. Ranking combined lets a
        # strongly one-sided ticker (e.g. all-call conviction) get allocated
        # accordingly instead of being forced into an artificial 50/50 split.
        actionable_signals = (
            scored_options_df
            .sort_values('confidence_score', ascending=False)
            .groupby('underlying_ticker')
            .head(3)
            .copy()
        )

        today_str = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        inserted_count = 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in actionable_signals.iterrows():
                signal_id = f"SIG_{uuid.uuid4().hex[:8].upper()}"
                entry_price = row.get('mark_price', 0.0)
                
                # We still drop zero-bid glitch contracts
                if entry_price <= 0:
                    continue
                    
                target_price = entry_price * (1 + self.profit_target_pct)
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)

                insert_sql = """
                INSERT INTO signals (
                    signal_id, underlying_ticker, option_symbol, option_type, 
                    expiration_date, strike_price, entry_date, entry_mark_price, 
                    target_price, stop_loss_price, confidence_score, status,
                    RSI_14, ATR_14, EMA_Alignment, Vol_OI_Ratio, Norm_Strike_Dist,
                    Delta, Gamma, Theta, Vega, impliedVolatility, IV_Rank, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                params = (
                    signal_id, row['underlying_ticker'], row['contractSymbol'], row['option_type'],
                    row['expiration_date'], row['strike'], today_str, round(entry_price, 2),
                    round(target_price, 2), round(stop_loss_price, 2), round(row['confidence_score'], 4), 'OPEN',
                    row.get('RSI_14'), row.get('ATR_14'), row.get('EMA_Alignment'), row.get('Vol_OI_Ratio'),
                    row.get('Norm_Strike_Dist'), row.get('Delta'), row.get('Gamma'), row.get('Theta'),
                    row.get('Vega'), row.get('impliedVolatility'), row.get('IV_Rank'), row.get('model_version', 'unknown')
                )
                
                try:
                    cursor.execute(insert_sql, params)
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Failed to insert signal {signal_id}: {e}")
            
            conn.commit()
            logger.info(f"Successfully saved {inserted_count} total signals to the database for ML training.")