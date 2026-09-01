import pandas as pd
import uuid
from datetime import datetime
import logging
from modules.storage_setup import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SignalGenerator")

class SignalGenerator:
    def __init__(self):
        self.db = DatabaseManager()
        self.profit_target_pct = 0.50
        self.stop_loss_pct = 0.30

        # --- TOP-N ALLOCATION RULES (for what actually gets traded) ---
        self.max_daily_trades = 5  # Total actionable trades to open per day across the whole market
        self.max_per_ticker = 1    # Max actionable trades per underlying to prevent piling into one stock

        # --- TICKER EXCLUSIONS ---
        # Exclude broad indices because their volatility profile doesn't match our +50%/-30% targets
        self.excluded_tickers = ['SPY', 'QQQ']

    def generate_and_store_signals(self, scored_options_df: pd.DataFrame) -> None:
        if scored_options_df.empty or 'confidence_score' not in scored_options_df.columns:
            return

        # 1. Filter out structurally incompatible tickers before doing any ranking.
        #    These are excluded from BOTH trading and training - SPY/QQQ have a
        #    volatility profile that doesn't fit the +50%/-30% target/stop, so
        #    there's no reason to keep learning from them either.
        filtered_df = scored_options_df[~scored_options_df['underlying_ticker'].isin(self.excluded_tickers)].copy()

        if filtered_df.empty:
            logger.info("No actionable signals remaining after applying ticker exclusions.")
            return

        # 2. Work out which rows qualify as "actionable" (the disciplined top-5,
        #    max-1-per-ticker picks) WITHOUT throwing away everything else.
        #    Everything in filtered_df still gets stored for training - only
        #    the top-5 diversified subset gets flagged as actionable for the
        #    paper-trade simulator / dashboard to act on.
        actionable_subset = (
            filtered_df
            .sort_values('confidence_score', ascending=False)
            .groupby('underlying_ticker')
            .head(self.max_per_ticker)
            .sort_values('confidence_score', ascending=False)
            .head(self.max_daily_trades)
        )
        actionable_symbols = set(actionable_subset['contractSymbol'])

        today_str = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        inserted_count = 0
        actionable_count = 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Store EVERY non-excluded signal, not just the top 5. The model
            # needs a broad, varied sample to train on - narrowing storage to
            # only the top picks would recreate the same feedback-loop problem
            # we already fixed once before.
            for _, row in filtered_df.iterrows():
                signal_id = f"SIG_{uuid.uuid4().hex[:8].upper()}"
                entry_price = row.get('mark_price', 0.0)

                # We still drop zero-bid glitch contracts
                if entry_price <= 0:
                    continue

                target_price = entry_price * (1 + self.profit_target_pct)
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)

                is_actionable = 1 if row['contractSymbol'] in actionable_symbols else 0

                insert_sql = """
                INSERT INTO signals (
                    signal_id, underlying_ticker, option_symbol, option_type,
                    expiration_date, strike_price, entry_date, entry_mark_price,
                    target_price, stop_loss_price, confidence_score, status,
                    RSI_14, ATR_14, EMA_Alignment, Vol_OI_Ratio, Norm_Strike_Dist,
                    Delta, Gamma, Theta, Vega, impliedVolatility, IV_Rank, model_version,
                    is_actionable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                params = (
                    signal_id, row['underlying_ticker'], row['contractSymbol'], row['option_type'],
                    row['expiration_date'], row['strike'], today_str, round(entry_price, 2),
                    round(target_price, 2), round(stop_loss_price, 2), round(row['confidence_score'], 4), 'OPEN',
                    row.get('RSI_14'), row.get('ATR_14'), row.get('EMA_Alignment'), row.get('Vol_OI_Ratio'),
                    row.get('Norm_Strike_Dist'), row.get('Delta'), row.get('Gamma'), row.get('Theta'),
                    row.get('Vega'), row.get('impliedVolatility'), row.get('IV_Rank'), row.get('model_version', 'unknown'),
                    is_actionable
                )

                try:
                    cursor.execute(insert_sql, params)
                    inserted_count += 1
                    actionable_count += is_actionable
                except Exception as e:
                    logger.error(f"Failed to insert signal {signal_id}: {e}")

            conn.commit()
            logger.info(
                f"Successfully saved {inserted_count} total signals to the database for ML training "
                f"({actionable_count} flagged actionable for paper trading)."
            )