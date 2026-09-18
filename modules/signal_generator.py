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
        self.max_daily_trades = 5  # Max actionable trades per day across the whole market
        self.max_per_ticker = 1    # Max actionable trades per underlying

        # --- QUALITY FLOOR ---
        # The system used to be FORCED to flag 5 trades every single day, even
        # on days where the model had no conviction about anything (observed
        # minimum confidence among forced picks: 0.000). Out-of-fold testing
        # showed this dragged win rate from ~49% (top picks only) down to ~38%.
        # "Trade nothing today" is a valid action and this floor allows it.
        #
        # Only applies to xgb_v1, whose scores are real probabilities. The
        # baseline_heuristic produces scores bounded 0.55-0.85 by construction
        # (a hand-tuned formula, not a probability), so the same numeric floor
        # would mean something completely different there.
        self.min_confidence_floor = 0.70
        self.floor_applies_to = {'xgb_v1'}

        # --- TICKER EXCLUSIONS ---
        self.excluded_tickers = ['SPY', 'QQQ']

    def generate_and_store_signals(self, scored_options_df: pd.DataFrame) -> None:
        if scored_options_df.empty or 'confidence_score' not in scored_options_df.columns:
            return

        # 1. Filter out structurally incompatible tickers before ranking.
        filtered_df = scored_options_df[~scored_options_df['underlying_ticker'].isin(self.excluded_tickers)].copy()

        if filtered_df.empty:
            logger.info("No actionable signals remaining after applying ticker exclusions.")
            return

        # 2. Determine which rows qualify as "actionable". Everything in
        #    filtered_df is still STORED for training - the floor and the
        #    top-N caps only control the is_actionable flag, never storage.
        candidates = filtered_df.copy()

        model_version = str(candidates['model_version'].iloc[0]) if 'model_version' in candidates.columns else 'unknown'
        floor_active = model_version in self.floor_applies_to

        if floor_active:
            pre_floor_count = len(candidates)
            best_available = candidates['confidence_score'].max()
            candidates = candidates[candidates['confidence_score'] >= self.min_confidence_floor]
            if candidates.empty:
                logger.info(
                    f"No signal cleared the {self.min_confidence_floor:.2f} quality floor today "
                    f"(best available: {best_available:.4f} out of {pre_floor_count} candidates). "
                    "Flagging zero actionable trades - sitting this day out."
                )
            else:
                logger.info(
                    f"{len(candidates)}/{pre_floor_count} candidates cleared the "
                    f"{self.min_confidence_floor:.2f} quality floor."
                )
        else:
            logger.info(
                f"Quality floor not applied for model_version='{model_version}' "
                "(floor is calibrated for xgb_v1 probabilities only)."
            )

        if candidates.empty:
            actionable_symbols = set()
        else:
            actionable_subset = (
                candidates
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

            # Store EVERY non-excluded signal, not just the actionable ones.
            # The model needs a broad, varied sample to train on.
            for _, row in filtered_df.iterrows():
                signal_id = f"SIG_{uuid.uuid4().hex[:8].upper()}"
                entry_price = row.get('mark_price', 0.0)

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