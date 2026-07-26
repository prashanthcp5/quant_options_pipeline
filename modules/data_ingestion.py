import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataIngestion")

class DataIngestionEngine:
    """
    Fetches underlying stock data and option chains, applying strict liquidity 
    and timeframe filters to remove noisy/illiquid contracts.
    """
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        # Hard filter thresholds
        self.min_oi = 500
        self.min_volume = 100
        self.max_spread_pct = 0.05  # 5%
        self.min_dte = 15
        self.max_dte = 45

    def get_filtered_options(self) -> pd.DataFrame:
        """
        Loops through tickers and expirations, fetches options, and applies filters.
        Returns a single unified DataFrame of all valid options.
        """
        all_options = []
        today = datetime.today().date()

        for ticker_symbol in self.tickers:
            logger.info(f"Fetching data for {ticker_symbol}...")
            ticker = yf.Ticker(ticker_symbol)
            
            try:
                expirations = ticker.options
            except Exception as e:
                logger.error(f"Failed to fetch expirations for {ticker_symbol}: {e}")
                continue

            for exp_date_str in expirations:
                # Calculate Days to Expiration (DTE)
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days

                # Filter 1: DTE must be between 15 and 45 days
                if not (self.min_dte <= dte <= self.max_dte):
                    continue

                try:
                    chain = ticker.option_chain(exp_date_str)
                    
                    # Process Calls
                    calls = chain.calls
                    calls['option_type'] = 'CALL'
                    
                    # Process Puts
                    puts = chain.puts
                    puts['option_type'] = 'PUT'
                    
                    # Combine calls and puts for this expiration
                    options_df = pd.concat([calls, puts], ignore_index=True)
                    options_df['underlying_ticker'] = ticker_symbol
                    options_df['expiration_date'] = exp_date_str
                    options_df['DTE'] = dte
                    
                    # Apply Liquidity Filters
                    filtered_df = self._apply_hard_filters(options_df)
                    
                    if not filtered_df.empty:
                        all_options.append(filtered_df)
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch chain for {ticker_symbol} expiring {exp_date_str}: {e}")
                    continue

        if not all_options:
            logger.warning("No options passed the filters.")
            return pd.DataFrame()

        final_df = pd.concat(all_options, ignore_index=True)
        logger.info(f"Successfully processed {len(final_df)} valid option contracts across all tickers.")
        return final_df

    def _apply_hard_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Volume, Open Interest, and Bid/Ask spread filters."""
        # Ensure we have the necessary columns before filtering
        required_cols = ['volume', 'openInterest', 'bid', 'ask']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        # Handle missing data
        df = df.fillna({'volume': 0, 'openInterest': 0, 'bid': 0, 'ask': 0})

        # Calculate Mark Price (midpoint of bid and ask)
        df['mark_price'] = (df['bid'] + df['ask']) / 2.0

        # Avoid division by zero in spread calculation
        # Spread % = (Ask - Bid) / Mark Price
        df['spread_pct'] = np.where(
            df['mark_price'] > 0,
            (df['ask'] - df['bid']) / df['mark_price'],
            float('inf')
        )

        # Apply strict rules
        mask = (
            (df['openInterest'] > self.min_oi) &
            (df['volume'] > self.min_volume) &
            (df['spread_pct'] < self.max_spread_pct)
        )
        
        return df[mask]

if __name__ == "__main__":
    # Test the engine with highly liquid tickers
    test_tickers = ["SPY", "AAPL"]
    engine = DataIngestionEngine(tickers=test_tickers)
    
    print(f"Fetching and filtering options for {test_tickers}...")
    valid_options = engine.get_filtered_options()
    
    if not valid_options.empty:
        # Show just a few relevant columns to verify
        display_cols = ['underlying_ticker', 'contractSymbol', 'option_type', 'strike', 'DTE', 'volume', 'openInterest', 'spread_pct']
        print("\nSample of valid contracts:")
        print(valid_options[display_cols].head())