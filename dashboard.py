import streamlit as st
import pandas as pd
from modules.storage_setup import DatabaseManager

# --- Page Config ---
st.set_page_config(
    page_title="Quant Options Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Database Connection ---
@st.cache_data(ttl=60) # Cache data for 60 seconds to prevent spamming the DB
def load_data():
    db = DatabaseManager()
    with db.get_connection() as conn:
        signals_df = pd.read_sql("SELECT * FROM signals", conn)
    return signals_df

def main():
    st.title("📈 Quantitative Options ML Pipeline")
    st.markdown("Live signal tracking and forward-testing dashboard.")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No signals found in the database. Run `python main.py` first.")
        return

    # --- Sidebar Metrics ---
    st.sidebar.header("System Metrics")
    
    total_signals = len(df)
    open_signals = len(df[df['status'] == 'OPEN'])
    won_signals = len(df[df['status'] == 'WON'])
    lost_signals = len(df[df['status'] == 'LOST'])
    
    total_closed = won_signals + lost_signals
    win_rate = (won_signals / total_closed * 100) if total_closed > 0 else 0.0
    
    st.sidebar.metric(label="Total Signals Generated", value=total_signals)
    st.sidebar.metric(label="Currently OPEN", value=open_signals)
    st.sidebar.metric(label="Win Rate", value=f"{win_rate:.1f}%")
    
    st.sidebar.divider()
    
    # --- Main Content: Active Trades ---
    st.subheader(f"🟢 Active 'OPEN' Signals ({open_signals})")
    
    active_df = df[df['status'] == 'OPEN'].copy()
    
    if not active_df.empty:
        # Format the dataframe for display
        display_df = active_df[[
            'entry_date', 'underlying_ticker', 'option_symbol', 'option_type',
            'expiration_date', 'strike_price', 'entry_mark_price', 
            'target_price', 'stop_loss_price', 'confidence_score'
        ]].sort_values(by='confidence_score', ascending=False)
        
        # Round confidence for readability
        display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "entry_mark_price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
                "target_price": st.column_config.NumberColumn("+50% Target", format="$%.2f"),
                "stop_loss_price": st.column_config.NumberColumn("-30% Stop", format="$%.2f"),
                "confidence_score": "ML Confidence"
            }
        )
    else:
        st.info("No active signals currently open.")
        
    st.divider()

    # --- Historical Performance ---
    st.subheader("📊 Closed Trade History")
    closed_df = df[df['status'] != 'OPEN'].copy()
    
    if not closed_df.empty:
        closed_display = closed_df[[
            'entry_date', 'underlying_ticker', 'option_symbol', 'option_type', 
            'status', 'entry_mark_price'
        ]].sort_values(by='entry_date', ascending=False)
        
        # Color code the status column
        def color_status(val):
            color = 'green' if val == 'WON' else 'red' if val == 'LOST' else 'gray'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(
            closed_display.style.map(color_status, subset=['status']),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No closed trades yet. The Forward Tester will update statuses as targets/stops are hit.")

if __name__ == "__main__":
    main()