import sqlite3
from modules.storage_setup import DatabaseManager

def upgrade_schema():
    db = DatabaseManager()
    features = [
        'RSI_14', 'ATR_14', 'EMA_Alignment', 'Vol_OI_Ratio', 
        'Norm_Strike_Dist', 'Delta', 'Gamma', 'Theta', 'Vega', 'impliedVolatility'
    ]
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        print("Upgrading database schema...")
        for feature in features:
            try:
                cursor.execute(f"ALTER TABLE signals ADD COLUMN {feature} REAL")
                print(f"Added column: {feature}")
            except sqlite3.OperationalError:
                print(f"Column already exists (or error): {feature}")
        conn.commit()
    print("Database upgrade complete!")

if __name__ == "__main__":
    upgrade_schema()