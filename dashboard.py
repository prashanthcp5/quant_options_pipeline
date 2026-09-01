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
    if 'model_version' not in df.columns:
        df['model_version'] = 'unknown'
    df['model_version'] = df['model_version'].fillna('unknown')
    return df

def main():
    st.title("📈 ML Quant Options Pipeline")
    st.markdown("---")

    df = load_data()
    fresh_cutoff = pd.Timestamp.now() - pd.Timedelta(hours=48)
    available_versions = ["xgb_v1", "baseline_heuristic"]
    
    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Paper Trade Simulator", 
        "🧠 Model Analytics", 
        "🎓 Options 101", 
        "🤖 Monte Carlo Optimizer"
    ])

    # ==========================================
    # TAB 1: PAPER TRADE SIMULATOR
    # ==========================================
    with tab1:
        st.header("🧮 Paper Trade Execution Calculator")
        st.markdown("Use this calculator to simulate individual trades. **Shows the Top 5 highest-conviction signals from the last 48 hours (excluding SPY/QQQ).**")
        
        st.warning("⚠️ **Calibration Warning:** The Kelly sizing calculator relies on the model's `confidence_score`. Because the current model is miscalibrated, Kelly sizing will be overly aggressive. Do not use for live capital until the XGBoost model is recalibrated.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            bankroll = st.number_input("Simulated Bankroll ($):", min_value=100, value=500, step=100, key="sim_bankroll")
        with col2:
            sim_version = st.selectbox("Signal Source:", available_versions, index=0, key="sim_version")
        with col3:
            kelly_fraction = st.selectbox("Kelly Fraction (Dampener):", [1.0, 0.5, 0.25, 0.125], index=2, format_func=lambda x: f"{x}x (1/{int(1/x)} Kelly)", help="Dial down the Kelly allocation to protect against uncalibrated probabilities.")
        
        open_df = df[df['status'] == 'OPEN'].copy()
        
        # 1. Filter by Model Version
        open_df = open_df[open_df['model_version'] == sim_version]
        
        # 2. Blacklist macro indices (SPY, QQQ)
        open_df = open_df[~open_df['underlying_ticker'].isin(['SPY', 'QQQ'])]
        
        # 3. 48-Hour Freshness Filter
        open_df = open_df[open_df['entry_date'] >= fresh_cutoff]
        
        # 4. Relative Threshold: Take Top 5 Highest Conviction Signals
        open_df = open_df.sort_values(by='confidence_score', ascending=False).head(5).copy()
        
        if not open_df.empty:
            b = 1.666 
            
            # Apply Kelly Fraction manually selected by user (defaults to 0.25 / Quarter-Kelly)
            open_df['raw_kelly'] = open_df['confidence_score'] - ((1 - open_df['confidence_score']) / b)
            open_df['raw_kelly'] = open_df['raw_kelly'].clip(lower=0) 
            open_df['Suggested Risk ($)'] = (bankroll * (open_df['raw_kelly'] * kelly_fraction))
            
            open_df['Contract Cost'] = open_df['entry_mark_price'] * 100
            open_df['Contracts to Buy'] = np.floor(open_df['Suggested Risk ($)'] / open_df['Contract Cost']).astype(int)
            
            open_df['Total Spend'] = open_df['Contracts to Buy'] * open_df['Contract Cost']
            open_df['Target Profit (+50%)'] = open_df['Total Spend'] * 0.50
            open_df['Max Risk (-30%)'] = open_df['Total Spend'] * 0.30
            
            display_df = open_df[[
                'entry_date', 'underlying_ticker', 'option_symbol', 'option_type', 'confidence_score', 
                'Contract Cost', 'Contracts to Buy', 'Total Spend', 
                'Target Profit (+50%)', 'Max Risk (-30%)'
            ]].copy()
            
            display_df['confidence_score'] = (display_df['confidence_score'] * 100).round(1).astype(str) + '%'
            
            currency_cols = ['Contract Cost', 'Total Spend', 'Target Profit (+50%)', 'Max Risk (-30%)']
            for col in currency_cols:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No fresh open signals found for `{sim_version}` in the last 48 hours.")

    # ==========================================
    # TAB 2: MODEL ANALYTICS
    # ==========================================
    with tab2:
        st.header("🧠 Machine Learning Performance")
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            timeframe = st.selectbox(
                "Select timeframe:", 
                ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
                index=0 
            )
        with filter_col2:
            tab2_versions = ["xgb_v1", "baseline_heuristic", "All Versions"]
            selected_version = st.selectbox("Select Model Version:", tab2_versions, index=0)
        
        filtered_df = df.copy()
        today = pd.Timestamp.now().normalize()
        start_date = today
        
        if timeframe == "Last 30 Days":
            start_date = today - pd.Timedelta(days=30)
            filtered_df = filtered_df[filtered_df['entry_date'] >= start_date]
        elif timeframe == "Last 90 Days":
            start_date = today - pd.Timedelta(days=90)
            filtered_df = filtered_df[filtered_df['entry_date'] >= start_date]
        elif timeframe == "Year to Date":
            start_date = pd.Timestamp(year=today.year, month=1, day=1)
            filtered_df = filtered_df[filtered_df['entry_date'] >= start_date]
        else:
            start_date = df['entry_date'].min() if not df.empty else today
            
        if selected_version != "All Versions":
            filtered_df = filtered_df[filtered_df['model_version'] == selected_version]
            
        st.caption(f"📅 **Data shown for:** {start_date.strftime('%b %d, %Y')} ➔ {today.strftime('%b %d, %Y')} | **Version:** `{selected_version}`")
            
        filtered_closed = filtered_df[filtered_df['status'].isin(['WON', 'LOST'])]
        
        # Isolate Non-Index Closed Trades >= 50% Confidence
        no_index_closed = filtered_closed[~filtered_closed['underlying_ticker'].isin(['SPY', 'QQQ'])] if not filtered_closed.empty else pd.DataFrame()
        top_tier_filtered = no_index_closed[no_index_closed['confidence_score'] >= 0.50] if not no_index_closed.empty else pd.DataFrame()

        # Metrics display with explanatory hover tooltips
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Generated Signals", 
            value=len(filtered_df),
            help="Total raw Call and Put signals logged by this model version over the selected date range, including open and closed positions."
        )
        
        if not top_tier_filtered.empty:
            top_wins = len(top_tier_filtered[top_tier_filtered['status'] == 'WON'])
            top_win_rate = (top_wins / len(top_tier_filtered)) * 100
            col2.metric(
                label="Top Tier Win Rate (Non-Index >=50%)", 
                value=f"{top_win_rate:.2f}%",
                help="Percentage of settled single-stock trades (excl. SPY/QQQ) with >=50% confidence that hit +50% profit target before hitting -30% stop-loss."
            )
            col3.metric(
                label="Top Tier Closed Trades", 
                value=len(top_tier_filtered),
                help="Total sample size of completed single-stock trades scoring >=50% confidence that reached a final status of WON or LOST."
            )
        else:
            col2.metric(
                label="Top Tier Win Rate", 
                value="N/A",
                help="No closed single-stock trades matching >=50% confidence were found for this selection."
            )
            col3.metric(
                label="Top Tier Closed Trades", 
                value=0,
                help="No closed single-stock trades matching >=50% confidence were found for this selection."
            )

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Feature Importance")
            model_path = Path("models/xgb_model.json")
            if model_path.exists():
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                features = ['Delta', 'RSI_14', 'Norm_Strike_Dist', 'ATR_14', 'IV_Rank']
                
                if len(model.feature_importances_) == len(features):
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=True)
                    
                    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color_continuous_scale='viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)

        with chart_col2:
            st.subheader(f"Model Calibration ({selected_version})")
            if not filtered_closed.empty and 'confidence_score' in filtered_closed.columns:
                calib_df = filtered_closed.copy()
                calib_df['Win'] = (calib_df['status'] == 'WON').astype(int)
                bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                labels = ['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
                calib_df['Confidence Bin'] = pd.cut(calib_df['confidence_score'], bins=bins, labels=labels, include_lowest=True)
                win_rates = calib_df.groupby('Confidence Bin', observed=False)['Win'].mean().reset_index()
                win_rates['Win'] = win_rates['Win'].fillna(0) * 100
                
                fig_calib = px.bar(win_rates, x='Confidence Bin', y='Win', color='Win', color_continuous_scale='RdYlGn')
                fig_calib.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_calib, use_container_width=True)
            else:
                st.info("No closed trades matching this filter.")

    # ==========================================
    # TAB 3: OPTIONS 101 GUIDE
    # ==========================================
    with tab3:
        st.header("🎓 Options 101: A Non-Finance Guide")
        st.markdown("Welcome to the quantitative pipeline. This guide explains how the system executes trades.")
        
        st.subheader("1. What is an Option?")
        st.markdown("An option is a contract that gives you the right to buy (Call) or sell (Put) 100 shares at a locked-in strike price before expiration.")

        st.subheader("2. Calls vs. Puts")
        st.markdown("* **Call:** Purchased when the model expects the price to move **UP**.\n* **Put:** Purchased when the model expects the price to move **DOWN**.")

        st.subheader("3. Lifecycle of a Trade")
        st.markdown("""
        * **Phase 1: Inception:** The AI logs actionable Call or Put signals to your database.
        * **Phase 2: Execution:** The top-ranked daily signals (excluding macro indices) are dynamically selected.
        * **Phase 3: Exit Target:** If the contract gains **+50%**, status updates to `WON`. If it falls **-30%**, status updates to `LOST`.
        """)
        
    # ==========================================
    # TAB 4: MONTE CARLO OPTIMIZER
    # ==========================================
    with tab4:
        st.header("🤖 Monte Carlo Portfolio Optimizer")
        st.markdown("Allocates capital across current top-conviction trades to maximize Expected Value.")
        
        st.warning("⚠️ **Calibration Warning:** EV projections rely on current confidence scores. Because the model requires recalibration, these EV figures are hypothetical. Treat output as an educational simulation.")
        
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        with opt_col1:
            opt_bankroll = st.number_input("Max Portfolio Budget ($):", min_value=100, value=500, step=50, key="opt_bankroll")
        with opt_col2:
            opt_version = st.selectbox("Signal Source:", available_versions, index=0, key="opt_version")
        with opt_col3:
            opt_kelly = st.selectbox("Kelly Fraction Constraint:", [1.0, 0.5, 0.25, 0.125], index=2, format_func=lambda x: f"{x}x (1/{int(1/x)} Kelly)", key="opt_kelly")
        
        mc_df = df[df['status'] == 'OPEN'].copy()
        
        # 1. Filter by Model Version
        mc_df = mc_df[mc_df['model_version'] == opt_version]
        
        # 2. Filter out SPY/QQQ and apply freshness cutoff
        mc_df = mc_df[
            (~mc_df['underlying_ticker'].isin(['SPY', 'QQQ'])) &
            (mc_df['entry_date'] >= fresh_cutoff)
        ].copy()
        
        # 3. Dynamic Relative Top 5 selection
        mc_df = mc_df.sort_values(by='confidence_score', ascending=False).head(5).copy()
        
        if not mc_df.empty:
            b = 1.666 
            mc_df['raw_kelly'] = mc_df['confidence_score'] - ((1 - mc_df['confidence_score']) / b)
            mc_df['raw_kelly'] = mc_df['raw_kelly'].clip(lower=0) 
            mc_df['Suggested Risk ($)'] = (opt_bankroll * (mc_df['raw_kelly'] * opt_kelly))
            mc_df['Contract Cost'] = mc_df['entry_mark_price'] * 100
            
            mc_df['EV'] = (0.50 * mc_df['Contract Cost'] * mc_df['confidence_score']) - (0.30 * mc_df['Contract Cost'] * (1 - mc_df['confidence_score']))
            
            if st.button("Run Simulation"):
                with st.spinner("Running 10,000 Monte Carlo combinations..."):
                    best_portfolio = []
                    max_ev = -1
                    best_spend = 0
                    
                    for _ in range(10000):
                        current_spend = 0
                        current_ev = 0
                        portfolio = []
                        
                        shuffled_options = mc_df.sample(frac=1).reset_index(drop=True)
                        
                        for idx, row in shuffled_options.iterrows():
                            max_qty_by_budget = (opt_bankroll - current_spend) // row['Contract Cost']
                            max_qty_by_kelly = int(np.floor(row['Suggested Risk ($)'] / row['Contract Cost']))
                            max_allowed = int(min(max_qty_by_budget, max_qty_by_kelly))
                            
                            if max_allowed > 0:
                                qty = np.random.randint(0, max_allowed + 1)
                                if qty > 0:
                                    spend = qty * row['Contract Cost']
                                    ev = qty * row['EV']
                                    current_spend += spend
                                    current_ev += ev
                                    portfolio.append({
                                        'Ticker': row['underlying_ticker'],
                                        'Option': row['option_symbol'], 
                                        'Type': row['option_type'],
                                        'Contracts': qty, 
                                        'Cost per Unit': f"${row['Contract Cost']:.2f}",
                                        'Total Allocation': spend, 
                                        'Added EV': ev
                                    })
                        
                        if current_ev > max_ev and current_spend <= opt_bankroll:
                            max_ev = current_ev
                            best_spend = current_spend
                            best_portfolio = portfolio
                    
                    if best_portfolio:
                        st.success("Optimization Complete! Processed 10,000 scenarios.")
                        sum_col1, sum_col2, sum_col3 = st.columns(3)
                        sum_col1.metric("Budget Utilized", f"${best_spend:.2f}", f"{((best_spend/opt_bankroll)*100):.1f}% of Capacity")
                        sum_col2.metric("Remaining Cash", f"${(opt_bankroll - best_spend):.2f}")
                        sum_col3.metric("Expected Portfolio EV", f"+${max_ev:.2f}")
                        
                        st.markdown("### 🛒 Optimized Execution Ticket")
                        port_df = pd.DataFrame(best_portfolio)
                        port_df['Total Allocation'] = port_df['Total Allocation'].apply(lambda x: f"${x:,.2f}")
                        port_df['Added EV'] = port_df['Added EV'].apply(lambda x: f"+${x:,.2f}")
                        st.dataframe(port_df, use_container_width=True, hide_index=True)
                    else:
                        st.error("Budget is too small to safely purchase any available contracts based on Kelly limits.")
        else:
            st.warning("⚠️ No fresh signals available to optimize. Check your data ingestion script.")

if __name__ == "__main__":
    main()