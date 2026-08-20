import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime
import logging
import re
from modules.storage_setup import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ForwardTester")

class ForwardTester:
    """
    Evaluates all OPEN signals in the database against live market prices.
    Updates statuses to WON, LOST, or EXPIRED to build historical accuracy data.
    """
    def __init__(self):
        self.db = DatabaseManager()

    def evaluate_open_signals(self) -> None:
        """
        Fetches all OPEN signals, gets live prices, and applies the rules:
        - If mark >= target: WON
        - If mark <= stop_loss: LOST
        - If today > expiration: EXPIRED
        """
        with self.db.get_connection() as conn:
            # 1. Fetch all OPEN signals
            open_signals_df = pd.read_sql("SELECT * FROM signals WHERE status = 'OPEN'", conn)
            
        if open_signals_df.empty:
            logger.info("No OPEN signals to evaluate.")
            return

        logger.info(f"Evaluating {len(open_signals_df)} OPEN signals...")
        today_date = datetime.today().date()
        today_str = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        updates = []
        evaluations = []

        for _, row in open_signals_df.iterrows():
            signal_id = row['signal_id']
            option_symbol = row['option_symbol']
            target = row['target_price']
            stop = row['stop_loss_price']
            exp_date = datetime.strptime(row['expiration_date'], '%Y-%m-%d').date()

            # 2. Check for expiration FIRST
            if today_date > exp_date:
                updates.append(( 'EXPIRED', signal_id ))
                logger.info(f"Signal {signal_id} ({option_symbol}) EXPIRED.")
                continue

            # 3. Fetch current live price from yfinance using Bid/Ask midpoint
            try:
                # Extract the underlying ticker from the database or the OCC option symbol
                underlying = row.get('underlying_ticker')
                if pd.isna(underlying) or not underlying:
                    match = re.match(r"^([A-Z]+)\d{6}[CP]\d{8}$", option_symbol)
                    underlying = match.group(1) if match else option_symbol

                ticker = yf.Ticker(underlying)
                exp_date_str = exp_date.strftime('%Y-%m-%d')
                
                # Fetch the live chain for this specific expiration
                chain = ticker.option_chain(exp_date_str)
                option_type = row.get('option_type', 'CALL')
                contracts = chain.puts if option_type == 'PUT' else chain.calls

                # Isolate our specific contract
                contract_data = contracts[contracts['contractSymbol'] == option_symbol]
                
                if contract_data.empty:
                    logger.warning(f"Contract {option_symbol} not found in live chain. Skipping.")
                    continue
                    
                current_bid = float(contract_data['bid'].iloc[0])
                current_ask = float(contract_data['ask'].iloc[0])
                
                # Calculate the true current mark price
                current_price = (current_bid + current_ask) / 2
                
                # Prevent evaluating on stale zero-bid liquidity glitches
                if current_price <= 0.01:
                    logger.warning(f"Contract {option_symbol} has zero-bid liquidity (Mark: ${current_price:.2f}). Skipping.")
                    continue
                    
            except Exception as e:
                logger.error(f"Error fetching live chain data for {option_symbol}: {e}")
                continue

            # 4. Log this check in our evaluations table
            evaluations.append((signal_id, today_str, round(current_price, 2), "Daily Check"))

            # 5. Check Win/Loss conditions
            new_status = 'OPEN'
            if current_price >= target:
                new_status = 'WON'
                logger.info(f"Signal {signal_id} WON! True Mark ${current_price:.2f} hit target ${target:.2f}.")
            elif current_price <= stop:
                new_status = 'LOST'
                logger.info(f"Signal {signal_id} LOST. True Mark ${current_price:.2f} hit stop ${stop:.2f}.")

            if new_status != 'OPEN':
                updates.append(( new_status, signal_id ))

        # 6. Save updates to the database
        self._commit_updates(updates, evaluations)

    def _commit_updates(self, updates: list, evaluations: list) -> None:
        """Executes the SQL updates transactionally."""
        if not updates and not evaluations:
            return

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update statuses
            if updates:
                cursor.executemany("UPDATE signals SET status = ? WHERE signal_id = ?", updates)
                logger.info(f"Updated status for {len(updates)} signals.")

            # Insert evaluation logs
            if evaluations:
                eval_sql = """
                INSERT INTO signal_evaluations (signal_id, check_timestamp, current_mark_price, notes)
                VALUES (?, ?, ?, ?)
                """
                cursor.executemany(eval_sql, evaluations)
                logger.info(f"Logged {len(evaluations)} price evaluations.")
                
            conn.commit()

    def print_performance_metrics(self) -> None:
        """Calculates and prints simple metrics from the database."""
        with self.db.get_connection() as conn:
            df = pd.read_sql("SELECT status, count(*) as cnt FROM signals GROUP BY status", conn)
            
        if df.empty:
            print("No signals in database.")
            return
            
        metrics = dict(zip(df['status'], df['cnt']))
        won = metrics.get('WON', 0)
        lost = metrics.get('LOST', 0)
        total_closed = won + lost
        
        print("\n=== SYSTEM PERFORMANCE ===")
        print(f"OPEN Signals:    {metrics.get('OPEN', 0)}")
        print(f"WON Signals:     {won}")
        print(f"LOST Signals:    {lost}")
        print(f"EXPIRED Signals: {metrics.get('EXPIRED', 0)}")
        
        if total_closed > 0:
            win_rate = (won / total_closed) * 100
            print(f"WIN RATE:        {win_rate:.2f}%")
        else:
            print("WIN RATE:        N/A (No closed trades yet)")
        print("==========================\n")

if __name__ == "__main__":
    tester = ForwardTester()
    tester.evaluate_open_signals()
    tester.print_performance_metrics()