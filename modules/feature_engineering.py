import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEngineering")

class FeatureEngineer:
    """
    Computes technical indicators for the underlying stock and calculates 
    Options Greeks and dynamics (Black-Scholes) for the options chain.
    """
    
    def __init__(self, risk_free_rate: float = 0.043):
        # 4.3% risk-free rate approximation (current Treasury yield context)
        self.r = risk_free_rate  

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

        return df

    def calculate_greeks(self, row: pd.Series) -> pd.Series:
        """
        Calculates Delta, Gamma, Theta, and Vega using the Black-Scholes model.
        """
        S = row['stock_price']
        K = row['strike']
        T = row['DTE'] / 365.0  # Time in years
        sigma = row['impliedVolatility']
        r = self.r
        opt_type = row['option_type']

        # Handle edge cases (expired or 0 vol)
        if T <= 0 or sigma <= 0 or S <= 0:
            return pd.Series({'Delta': 0.0, 'Gamma': 0.0, 'Theta': 0.0, 'Vega': 0.0})

        # Black-Scholes d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Common terms
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_prime_d1 = norm.pdf(d1)

        gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
        vega = S * N_prime_d1 * np.sqrt(T) / 100  # Per 1% change

        if opt_type == 'CALL':
            delta = N_d1
            theta = (- (S * sigma * N_prime_d1) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * N_d2) / 365
        else: # PUT
            delta = N_d1 - 1
            theta = (- (S * sigma * N_prime_d1) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        return pd.Series({'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega})

    def process_features(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges underlying technicals and calculates option-specific features.
        """
        if options_df.empty:
            return options_df

        tickers = options_df['underlying_ticker'].unique()
        stock_data_map = {}

        # 1. Fetch and calculate stock technicals
        for ticker in tickers:
            logger.info(f"Processing technicals for {ticker}...")
            hist_df = self._get_historical_stock_data(ticker)
            if not hist_df.empty:
                hist_df = self._calculate_technicals(hist_df)
                # Get the latest row for current metrics
                latest = hist_df.iloc[-1]
                stock_data_map[ticker] = {
                    'stock_price': latest['Close'],
                    'RSI_14': latest['RSI_14'],
                    'ATR_14': latest['ATR_14'],
                    'EMA_Alignment': latest['EMA_Alignment']
                }

        # 2. Map stock metrics to the options dataframe
        # Creating a DataFrame from the map to merge easily
        stock_metrics_df = pd.DataFrame.from_dict(stock_data_map, orient='index').reset_index()
        stock_metrics_df.rename(columns={'index': 'underlying_ticker'}, inplace=True)
        
        df = pd.merge(options_df, stock_metrics_df, on='underlying_ticker', how='left')

        # 3. Calculate Option Dynamics
        logger.info("Calculating Options Dynamics and Greeks...")
        
        # Volume/OI Ratio (Unusual activity indicator)
        df['Vol_OI_Ratio'] = df['volume'] / df['openInterest']
        
        # Normalized Strike Distance (How many ATRs away is the strike?)
        # (Stock - Strike) / ATR. Positive = ITM for Calls, Negative = OTM for calls.
        df['Norm_Strike_Dist'] = (df['stock_price'] - df['strike']) / df['ATR_14']

        # Calculate Greeks vector-wise via apply
        greeks = df.apply(self.calculate_greeks, axis=1)
        df = pd.concat([df, greeks], axis=1)

        # Drop rows with NaN (from missing historical data or calculation errors)
        df.dropna(subset=['Delta', 'RSI_14', 'ATR_14'], inplace=True)
        
        logger.info(f"Feature engineering complete. Dataset shape: {df.shape}")
        return df

if __name__ == "__main__":
    # To test this, we'll import the ingestion engine, grab data, and run it through features
    from data_ingestion import DataIngestionEngine
    
    test_tickers = ["AAPL"]
    ingestion = DataIngestionEngine(tickers=test_tickers)
    raw_options = ingestion.get_filtered_options()
    
    if not raw_options.empty:
        engineer = FeatureEngineer()
        featured_options = engineer.process_features(raw_options)
        
        display_cols = ['contractSymbol', 'stock_price', 'strike', 'RSI_14', 'Norm_Strike_Dist', 'Delta', 'Vol_OI_Ratio']
        print("\nFeature Engineered Data Sample:")
        print(featured_options[display_cols].head())