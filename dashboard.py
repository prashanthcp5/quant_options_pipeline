import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import xgboost as xgb
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Quant Pipeline Dashboard", layout="wide", page_icon="📈")

def load_data():
    conn = sqlite3.connect("data/options_pipeline.db")
    df = pd.read_sql("SELECT * FROM signals", conn)
    conn.close()
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    return df

def main():
    st.title("📈 ML Quant Options Pipeline")
    st.markdown("---")

    df = load_data()
    closed_df = df[df['status'].isin(['WON', 'LOST'])]
    
    # Define Goldilocks dataframe globally for analytics
    if not closed_df.empty and 'confidence_score' in closed_df.columns:
        goldilocks_closed = closed_df[(closed_df['confidence_score'] >= 0.60) & (closed_df['confidence_score'] <= 0.80)]
    else:
        goldilocks_closed = pd.DataFrame()

    # --- UI OPTIMIZATION: TABS ---
    tab1, tab2, tab3 = st.tabs(["🚀 Paper Trade Simulator", "🧠 Model Analytics", "🎓 Options 101"])

    # ==========================================
    # TAB 1: PAPER TRADE SIMULATOR
    # ==========================================
    with tab1:
        st.header("🧮 Paper Trade Execution Calculator")
        st.markdown("Use this calculator to simulate trades. Write down the outputted contracts and track if they hit the +50% target or -30% stop loss.")
        
        bankroll = st.number_input("Enter your Simulated Bankroll ($):", min_value=100, value=10000, step=100)
        
        open_df = df[df['status'] == 'OPEN'].sort_values(by='confidence_score', ascending=False)
        open_df = open_df[(open_df['confidence_score'] >= 0.60) & (open_df['confidence_score'] <= 0.80)].copy()
        
        if not open_df.empty:
            b = 1.666 
            
            # Math execution
            open_df['raw_kelly'] = open_df['confidence_score'] - ((1 - open_df['confidence_score']) / b)
            open_df['raw_kelly'] = open_df['raw_kelly'].clip(lower=0) 
            open_df['Suggested Risk ($)'] = (bankroll * (open_df['raw_kelly'] / 4))
            
            open_df['Contract Cost'] = open_df['entry_mark_price'] * 100
            open_df['Contracts to Buy'] = np.floor(open_df['Suggested Risk ($)'] / open_df['Contract Cost']).astype(int)
            
            open_df['Total Spend'] = open_df['Contracts to Buy'] * open_df['Contract Cost']
            open_df['Target Profit (+50%)'] = open_df['Total Spend'] * 0.50
            open_df['Max Risk (-30%)'] = open_df['Total Spend'] * 0.30
            
            display_df = open_df[[
                'entry_date', 'underlying_ticker', 'option_symbol', 'confidence_score', 
                'Contract Cost', 'Contracts to Buy', 'Total Spend', 
                'Target Profit (+50%)', 'Max Risk (-30%)'
            ]].copy()
            
            display_df['confidence_score'] = (display_df['confidence_score'] * 100).round(1).astype(str) + '%'
            
            currency_cols = ['Contract Cost', 'Total Spend', 'Target Profit (+50%)', 'Max Risk (-30%)']
            for col in currency_cols:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open signals currently in the Goldilocks zone. Check back tomorrow.")

    # ==========================================
    # TAB 2: MODEL ANALYTICS
    # ==========================================
    with tab2:
        st.header("🧠 Machine Learning Performance")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Generated Signals", len(df))
        if not goldilocks_closed.empty:
            gold_wins = len(goldilocks_closed[goldilocks_closed['status'] == 'WON'])
            gold_win_rate = (gold_wins / len(goldilocks_closed)) * 100
            col2.metric("Goldilocks Zone Win Rate", f"{gold_win_rate:.2f}%")
            col3.metric("Goldilocks Closed Trades", len(goldilocks_closed))
        else:
            col2.metric("Goldilocks Zone Win Rate", "N/A")
            col3.metric("Goldilocks Closed Trades", 0)

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Feature Importance")
            model_path = Path("models/xgb_model.json")
            if model_path.exists():
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                features = ['Delta', 'RSI_14', 'Norm_Strike_Dist', 'ATR_14', 'impliedVolatility']
                
                if len(model.feature_importances_) == len(features):
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=True)
                    
                    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color_continuous_scale='viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)

        with chart_col2:
            st.subheader("Model Calibration")
            if not closed_df.empty and 'confidence_score' in closed_df.columns:
                calib_df = closed_df.copy()
                calib_df['Win'] = (calib_df['status'] == 'WON').astype(int)
                bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
                calib_df['Confidence Bin'] = pd.cut(calib_df['confidence_score'], bins=bins, labels=labels, include_lowest=True)
                win_rates = calib_df.groupby('Confidence Bin', observed=False)['Win'].mean().reset_index()
                win_rates['Win'] = win_rates['Win'].fillna(0) * 100
                
                fig_calib = px.bar(win_rates, x='Confidence Bin', y='Win', color='Win', color_continuous_scale='RdYlGn')
                fig_calib.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_calib, use_container_width=True)

        if not df.empty and 'confidence_score' in df.columns:
            st.markdown("---")
            st.subheader("📉 Macro Trend: Confidence vs. Win Rate")
            trend_df = df.dropna(subset=['confidence_score']).copy()
            trend_df['Week'] = trend_df['entry_date'].dt.to_period('W').dt.start_time
            weekly_conf = trend_df.groupby('Week')['confidence_score'].mean().reset_index()
            weekly_conf['Avg Confidence (%)'] = (weekly_conf['confidence_score'] * 100).round(2)
            
            if not closed_df.empty:
                closed_trend_df = closed_df.copy()
                closed_trend_df['Week'] = closed_trend_df['entry_date'].dt.to_period('W').dt.start_time
                closed_trend_df['Win'] = (closed_trend_df['status'] == 'WON').astype(int)
                weekly_win = closed_trend_df.groupby('Week')['Win'].mean().reset_index()
                weekly_win['Win Rate (%)'] = (weekly_win['Win'] * 100).round(2)
                weekly_trend = pd.merge(weekly_conf[['Week', 'Avg Confidence (%)']], weekly_win[['Week', 'Win Rate (%)']], on='Week', how='left')
            else:
                weekly_trend = weekly_conf[['Week', 'Avg Confidence (%)']]
                weekly_trend['Win Rate (%)'] = 0.0

            fig_trend = px.line(weekly_trend, x='Week', y=['Avg Confidence (%)', 'Win Rate (%)'], markers=True)
            fig_trend.data[0].line.color = '#00CC96' 
            if len(fig_trend.data) > 1:
                fig_trend.data[1].line.color = '#FFA15A' 
            fig_trend.update_layout(yaxis_range=[0, 100], legend_title_text='')
            st.plotly_chart(fig_trend, use_container_width=True)

    # ==========================================
    # TAB 3: OPTIONS 101 GUIDE
    # ==========================================
    with tab3:
        st.header("🎓 Options 101: A Non-Finance Guide")
        st.markdown("Welcome to the quantitative pipeline. If you have never traded before, this guide will explain exactly what the AI is doing.")
        
        st.subheader("1. What is an Option?")
        st.markdown("""
        Think of an option like a **coupon**. 
        If you have a coupon to buy a TV for $500, but the TV's price at Best Buy suddenly jumps to $1,000, your coupon just became extremely valuable. Other people will want to buy that coupon from you.
        
        In the stock market, an option is a "coupon" that gives you the right to buy 100 shares of a stock at a locked-in price (called the **Strike Price**) before a specific expiration date.
        """)

        st.subheader("2. What is a 'Call'?")
        st.markdown("""
        There are two types of options: Calls and Puts. 
        * **Call Option:** You buy this when you think the stock price will go **UP**.
        * **Put Option:** You buy this when you think the stock price will go **DOWN**. 
        * *Note: Your AI currently only looks for Call Options.*
        """)

        st.subheader("3. The Lifecycle of a Trade")
        st.markdown("""
        Here is exactly how a trade flows from the AI to your bank account:
        
        * **Phase 1: Inception (The AI Scan)**
          The AI scans thousands of stocks. It finds a stock (like SOFI) that its math indicates is about to surge upward. It logs a specific Call Option to your dashboard.
        * **Phase 2: Buying the Contract**
          You look at the dashboard. It tells you the contract costs $45. You go to your broker, search the exact `option_symbol`, and buy 1 contract for $45. You do *not* own 100 shares of SOFI; you only own the coupon.
        * **Phase 3: The Wait (Price Movement)**
          The stock market opens. As the actual price of SOFI goes up, the "coupon" you hold becomes more valuable. The $45 you spent might now be worth $60.
        * **Phase 4: Selling (Closing the Trade)**
          You do not wait for the expiration date, and you never actually buy the 100 shares. Once the value of your coupon goes up by **+50%**, you click "Sell to Close" in your broker. You sell the coupon to someone else, and you lock in your profit. If the stock drops and the coupon loses **-30%** of its value, you sell it immediately to cut your losses.
        """)
        
        st.info("💡 **Why 100 Shares?** By law, one option contract represents 100 shares of stock. That is why an option listed at $0.45 on your broker's screen will actually cost you $45.00 to buy.")

if __name__ == "__main__":
    main()