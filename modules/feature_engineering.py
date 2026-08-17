import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEngineering")

class FeatureEngineer:
    """
    Computes technical indicators for the underlying stock, transforms Volatility into IV Rank,
    and calculates Options Greeks (Black-Scholes-Merton) accounting for continuous dividend yields.
    """
    
    def __init__(self, risk_free_rate: float = 0.043, dividend_yield: float = 0.015):
        # 4.3% risk-free rate approximation
        self.r = risk_free_rate  
        # 1.5% average dividend yield approximation (Merton extension)
        self.q = dividend_yield

    def _get_historical_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetches historical daily data for technical indicators."""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df

    def _calculate_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates RSI(14), ATR(14), and EMAs(20, 50, 200)."""
        if len(df) < 200:
            logger.warning("Not enough data to calculate 200 EMA.")
            
        # EMAs
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # EMA Alignment (1 if 20 > 50 > 200, -1 if 20 < 50 < 200, 0 otherwise)
        df['EMA_Alignment'] = np.where(
            (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_200']), 1,
            np.where((df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_200']), -1, 0)
        )

        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # ATR (14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR_14'] = true_range.rolling(window=14).mean()
        
        # Calculate Historical Volatility (252 trading days) to help establish a baseline
        df['Daily_Return'] = df['Close'].pct_change()
        df['Historical_Vol'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)

        return df

    def calculate_greeks(self, row: pd.Series) -> pd.Series:
        """
        Calculates Delta, Gamma, Theta, and Vega using the Black-Scholes-Merton model 
        (which accounts for continuous dividend yield 'q').
        """
        S = row['stock_price']
        K = row['strike']
        T = row['DTE'] / 365.0  # Time in years
        sigma = row['impliedVolatility']
        r = self.r
        q = self.q # Dividend Yield added!
        opt_type = row['option_type']

        # Handle edge cases (expired or 0 vol)
        if T <= 0 or sigma <= 0 or S <= 0:
            return pd.Series({'Delta': 0.0, 'Gamma': 0.0, 'Theta': 0.0, 'Vega': 0.0})

        # Black-Scholes-Merton d1 and d2 (Updated with -q)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Common terms
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_prime_d1 = norm.pdf(d1)

        # Greeks updated with the continuous dividend yield discount e^(-qT)
        gamma = (np.exp(-q * T) * N_prime_d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * N_prime_d1 * np.sqrt(T) / 100  # Per 1% change

        if opt_type == 'CALL':
            delta = np.exp(-q * T) * N_d1
            theta = (- (S * sigma * np.exp(-q * T) * N_prime_d1) / (2 * np.sqrt(T)) 
                     + q * S * np.exp(-q * T) * N_d1 
                     - r * K * np.exp(-r * T) * N_d2) / 365
        else: # PUT
            delta = np.exp(-q * T) * (N_d1 - 1)
            theta = (- (S * sigma * np.exp(-q * T) * N_prime_d1) / (2 * np.sqrt(T)) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        return pd.Series({'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega})

    def calculate_iv_rank(self, options_df: pd.DataFrame, stock_history_map: dict) -> pd.DataFrame:
        """
        Transforms raw Implied Volatility into IV Rank (0 to 100 scale) based on 
        the stock's 1-year historical volatility high/low baseline.
        """
        def get_ivr(row):
            ticker = row['underlying_ticker']
            current_iv = row['impliedVolatility']
            
            if ticker in stock_history_map and current_iv > 0:
                hist_vol = stock_history_map[ticker]['Historical_Vol'].dropna()
                if not hist_vol.empty:
                    iv_low = hist_vol.min()
                    iv_high = hist_vol.max()
                    
                    if iv_high > iv_low:
                        # IV Rank Formula
                        ivr = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
                        return max(0.0, min(100.0, ivr)) # Cap between 0 and 100
            return 50.0 # Safe neutral default if missing data
            
        options_df['IV_Rank'] = options_df.apply(get_ivr, axis=1)
        return options_df

    def process_features(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges underlying technicals, transforms IV, and calculates option-specific features.
        """
        if options_df.empty:
            return options_df

        tickers = options_df['underlying_ticker'].unique()
        stock_data_map = {}
        stock_history_map = {} # New map to hold raw series for IV Rank

        # 1. Fetch and calculate stock technicals
        for ticker in tickers:
            logger.info(f"Processing technicals for {ticker}...")
            hist_df = self._get_historical_stock_data(ticker)
            if not hist_df.empty:
                hist_df = self._calculate_technicals(hist_df)
                stock_history_map[ticker] = hist_df # Save full history for IVR
                
                # Get the latest row for current metrics
                latest = hist_df.iloc[-1]
                stock_data_map[ticker] = {
                    'stock_price': latest['Close'],
                    'RSI_14': latest['RSI_14'],
                    'ATR_14': latest['ATR_14'],
                    'EMA_Alignment': latest['EMA_Alignment']
                }

        # 2. Map stock metrics to the options dataframe
        stock_metrics_df = pd.DataFrame.from_dict(stock_data_map, orient='index').reset_index()
        stock_metrics_df.rename(columns={'index': 'underlying_ticker'}, inplace=True)
        df = pd.merge(options_df, stock_metrics_df, on='underlying_ticker', how='left')

        # 3. Calculate IV Rank (NEW!)
        logger.info("Transforming Volatility into IV Rank...")
        df = self.calculate_iv_rank(df, stock_history_map)

        # 4. Calculate Option Dynamics
        logger.info("Calculating Options Dynamics and Greeks...")
        df['Vol_OI_Ratio'] = df['volume'] / df['openInterest']
        df['Norm_Strike_Dist'] = (df['stock_price'] - df['strike']) / df['ATR_14']

        # Calculate Greeks vector-wise via apply
        greeks = df.apply(self.calculate_greeks, axis=1)
        df = pd.concat([df, greeks], axis=1)

        # Drop rows with NaN 
        df.dropna(subset=['Delta', 'RSI_14', 'ATR_14'], inplace=True)
        
        logger.info(f"Feature engineering complete. Dataset shape: {df.shape}")
        return df

if __name__ == "__main__":
    from data_ingestion import DataIngestionEngine
    
    test_tickers = ["AAPL"]
    ingestion = DataIngestionEngine(tickers=test_tickers)
    raw_options = ingestion.get_filtered_options()
    
    if not raw_options.empty:
        engineer = FeatureEngineer()
        featured_options = engineer.process_features(raw_options)
        
        display_cols = ['contractSymbol', 'strike', 'impliedVolatility', 'IV_Rank', 'Delta']
        print("\nFeature Engineered Data Sample:")
        print(featured_options[display_cols].head())