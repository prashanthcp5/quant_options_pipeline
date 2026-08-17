import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import xgboost as xgb
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
                
                with st.expander("❓ What do these terms mean?"):
                    st.markdown("""
                    **This chart shows what the AI cares about most when picking trades.** A bigger yellow bar means the AI relies heavily on this metric.
                    *   **Delta:** Measures how closely the option's price moves with the stock. 
                    *   **RSI_14 (Momentum):** A stock's "speedometer". It tells us if a stock is overbought (too hot) or oversold (too cold). 
                    *   **Norm_Strike_Dist:** How far our target price is from the stock's current price. 
                    *   **ATR_14 (Volatility):** The stock's typical daily price swing. 
                    *   **impliedVolatility (IV):** The market's expectation of future craziness. 
                    """)
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
            
            with st.expander("❓ How to read this chart"):
                st.markdown("""
                **This chart checks if the AI is "honest" or overconfident.** 
                *   **X-Axis (Bottom):** The AI's internal confidence level when it bought the trade.
                *   **Y-Axis (Left):** The *actual* win rate of those trades in real life.
                *   **Favorable:** You want the bars to get taller and greener as you move to the right. 
                *   **Warning:** Short red bars on the far right indicate "Market Traps" where the AI is overconfident. We now filter these out.
                """)
        else:
            st.info("Not enough feature-rich closed trades to plot calibration.")

    st.markdown("---")
    st.header("📋 Actionable Signals & Position Sizing")
    
    bankroll = st.number_input("Enter your total trading bankroll ($):", min_value=100, value=10000, step=100)
    
    open_df = df[df['status'] == 'OPEN'].sort_values(by='confidence_score', ascending=False)
    
    # Filter dashboard to only show Goldilocks setups
    open_df = open_df[(open_df['confidence_score'] >= 0.60) & (open_df['confidence_score'] <= 0.80)].copy()
    
    if not open_df.empty:
        # Reward-to-Risk ratio (50% target / 30% stop)
        b = 1.666 
        
        # Calculate raw Kelly Percentage: p - ((1-p)/b)
        open_df['raw_kelly'] = open_df['confidence_score'] - ((1 - open_df['confidence_score']) / b)
        open_df['raw_kelly'] = open_df['raw_kelly'].clip(lower=0) 
        
        # Calculate the 1/4 Fractional Kelly allocation for conservative risk
        open_df['Suggested Risk ($)'] = (bankroll * (open_df['raw_kelly'] / 4)).round(2)
        
        display_df = open_df[['entry_date', 'underlying_ticker', 'option_symbol', 'confidence_score', 'Suggested Risk ($)', 'entry_mark_price', 'target_price', 'stop_loss_price']].copy()
        display_df['confidence_score'] = (display_df['confidence_score'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open signals currently in the Goldilocks zone.")

if __name__ == "__main__":
    main()