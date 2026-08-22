import sqlite3
import pandas as pd

def check_directional_bias():
    conn = sqlite3.connect('data/options_pipeline.db')
    query = """
    SELECT 
        option_type, 
        count(*) as total_trades, 
        sum(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins,
        sum(CASE WHEN status='LOST' THEN 1 ELSE 0 END) as losses,
        sum(CASE WHEN status='WON' THEN 1 ELSE 0 END)*100.0/count(*) as win_rate_pct 
    FROM signals 
    WHERE model_version='xgb_v1' AND status IN ('WON', 'LOST') 
    GROUP BY option_type
    """
    df = pd.read_sql(query, conn)
    print("\n=== XGBoost v1 Directional Bias Check ===")
    if df.empty:
        print("No closed trades found for xgb_v1 yet.")
    else:
        print(df.to_string(index=False))
    print("=========================================\n")
    conn.close()

if __name__ == "__main__":
    check_directional_bias()