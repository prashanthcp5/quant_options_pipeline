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
    
    total_signals = len(df)
    open_signals = len(df[df['status'] == 'OPEN'])
    closed_df = df[df['status'].isin(['WON', 'LOST'])]
    
    win_rate = 0
    if not closed_df.empty:
        wins = len(closed_df[closed_df['status'] == 'WON'])
        win_rate = (wins / len(closed_df)) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals", total_signals)
    col2.metric("Currently OPEN", open_signals)
    col3.metric("Closed Trades", len(closed_df))
    col4.metric("Win Rate", f"{win_rate:.2f}%")

    st.markdown("---")
    st.header("🧠 Machine Learning Insights")
    
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("What Drives the Edge? (Feature Importance)")
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
                
                fig_imp = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig_imp.update_layout(showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("Feature mismatch. Ensure the model has been trained on the pruned features.")
        else:
            st.info("No trained ML model found.")

    with chart_col2:
        st.subheader("Model Calibration (Confidence vs. Actual Win Rate)")
        if not closed_df.empty and 'confidence_score' in closed_df.columns:
            calib_df = closed_df.copy()
            calib_df['Win'] = (calib_df['status'] == 'WON').astype(int)
            
            bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
            
            calib_df['Confidence Bin'] = pd.cut(calib_df['confidence_score'], bins=bins, labels=labels, include_lowest=True)
            
            win_rates_by_bin = calib_df.groupby('Confidence Bin', observed=False)['Win'].mean().reset_index()
            win_rates_by_bin['Win'] = win_rates_by_bin['Win'].fillna(0) * 100
            
            fig_calib = px.bar(
                win_rates_by_bin, 
                x='Confidence Bin', 
                y='Win',
                color='Win',
                color_continuous_scale='RdYlGn'
            )
            fig_calib.update_layout(yaxis_title="Actual Win Rate (%)", yaxis_range=[0, 100])
            fig_calib.add_hline(y=win_rate, line_dash="dot", annotation_text=f"Baseline ({win_rate:.1f}%)", annotation_position="bottom right")
            
            st.plotly_chart(fig_calib, use_container_width=True)
        else:
            st.info("Not enough feature-rich closed trades to plot calibration.")

    if not df.empty and 'confidence_score' in df.columns:
        st.markdown("---")
        st.subheader("📉 Macro Trend: AI Confidence vs. Actual Win Rate")
        
        trend_df = df.dropna(subset=['confidence_score']).copy()
        trend_df['Week'] = trend_df['entry_date'].dt.to_period('W').dt.start_time
        weekly_conf = trend_df.groupby('Week')['confidence_score'].mean().reset_index()
        weekly_conf['Avg Confidence (%)'] = (weekly_conf['confidence_score'] * 100).round(2)
        
        closed_trend_df = df[df['status'].isin(['WON', 'LOST'])].copy()
        if not closed_trend_df.empty:
            closed_trend_df['Week'] = closed_trend_df['entry_date'].dt.to_period('W').dt.start_time
            closed_trend_df['Win'] = (closed_trend_df['status'] == 'WON').astype(int)
            weekly_win = closed_trend_df.groupby('Week')['Win'].mean().reset_index()
            weekly_win['Win Rate (%)'] = (weekly_win['Win'] * 100).round(2)
            
            weekly_trend = pd.merge(weekly_conf[['Week', 'Avg Confidence (%)']], weekly_win[['Week', 'Win Rate (%)']], on='Week', how='left')
        else:
            weekly_trend = weekly_conf[['Week', 'Avg Confidence (%)']]
            weekly_trend['Win Rate (%)'] = 0.0

        fig_trend = px.line(
            weekly_trend,
            x='Week',
            y=['Avg Confidence (%)', 'Win Rate (%)'],
            markers=True,
            labels={'value': 'Percentage (%)', 'variable': 'Metric'}
        )
        
        fig_trend.data[0].line.color = '#00CC96' 
        if len(fig_trend.data) > 1:
            fig_trend.data[1].line.color = '#FFA15A' 
            
        fig_trend.update_layout(yaxis_range=[0, 100], legend_title_text='')
        st.plotly_chart(fig_trend, use_container_width=True)

    # ==========================================
    # --- UPGRADED TRADE EXECUTION CALCULATOR ---
    # ==========================================
    st.markdown("---")
    st.header("🧮 Trade Execution Calculator")
    
    bankroll_help = "Your total account balance. The AI uses this to calculate exactly how many contracts you can safely afford."
    bankroll = st.number_input("Enter your Total Trading Bankroll ($):", min_value=100, value=200, step=100, help=bankroll_help)
    
    with st.expander("❓ How does this Calculator work?"):
        st.markdown("""
        This calculator bridges the gap between AI math and actual broker execution.
        * **Contract Cost:** Options are bought in bundles of 100 shares. A $0.50 option costs $50 in real life.
        * **Contracts to Buy:** The AI takes your Bankroll, runs the Kelly formula to find your safe risk limit, and divides it by the Contract Cost. 
        * **What if it says '0'?** If it suggests 0 contracts, it means your bankroll is currently too small to buy even 1 contract of this setup without taking on mathematically reckless risk.
        * **Target Profit & Max Risk:** Exactly how many real dollars you will make (+50%) or lose (-30%) if the trade hits its triggers.
        """)
    
    open_df = df[df['status'] == 'OPEN'].sort_values(by='confidence_score', ascending=False)
    open_df = open_df[(open_df['confidence_score'] >= 0.60) & (open_df['confidence_score'] <= 0.80)].copy()
    
    if not open_df.empty:
        b = 1.666 
        
        # 1. Theoretical Math
        open_df['raw_kelly'] = open_df['confidence_score'] - ((1 - open_df['confidence_score']) / b)
        open_df['raw_kelly'] = open_df['raw_kelly'].clip(lower=0) 
        open_df['Suggested Risk ($)'] = (bankroll * (open_df['raw_kelly'] / 4))
        
        # 2. Real-World Broker Execution Math
        open_df['Contract Cost'] = open_df['entry_mark_price'] * 100
        
        # Using numpy floor to safely cast to integer contracts (rounding down)
        open_df['Contracts to Buy'] = np.floor(open_df['Suggested Risk ($)'] / open_df['Contract Cost']).astype(int)
        
        open_df['Total Spend'] = open_df['Contracts to Buy'] * open_df['Contract Cost']
        open_df['Target Profit (+50%)'] = open_df['Total Spend'] * 0.50
        open_df['Max Risk (-30%)'] = open_df['Total Spend'] * 0.30
        
        # 3. Formatting for the Dashboard
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
        st.info("No open signals currently in the Goldilocks zone.")

if __name__ == "__main__":
    main()