"""
TraderJoes v2.1.3 - Streamlit Interface
Second UI interface alongside PyQt - NO TRADING LOGIC CHANGES
ALL VALUES COME DIRECTLY FROM trader_service
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import json
from pathlib import Path

# Import the headless service
from trader_service import TraderJoesService

# Page config
st.set_page_config(
    page_title="TraderJoes v2.1.3",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="metric-container"] {
        background-color: #262626;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    
    # Initialize service in session state
    if "trader_service" not in st.session_state:
        st.session_state.trader_service = TraderJoesService("config.json")
    
    service = st.session_state.trader_service
    
    # Title
    st.title("🎯 TraderJoes v2.1.3 - Futures Trading Bot")
    st.caption("Professional Crypto Futures Trading with ML Optimization")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Control Panel")
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start Trading", use_container_width=True, disabled=service.is_trading()):
            service.start()
            st.success("Trading started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Trading", use_container_width=True, disabled=not service.is_trading()):
            service.stop()
            st.warning("Trading stopped!")
            st.rerun()
    
    # Trading status
    st.sidebar.divider()
    if service.is_trading():
        st.sidebar.success("🟢 TRADING ACTIVE")
    else:
        st.sidebar.error("🔴 TRADING STOPPED")
    
    # Page navigation
    st.sidebar.divider()
    page = st.sidebar.radio(
        "View",
        ["Active Trades", "Closed Trades"]
    )
    
    # Auto-refresh
    if service.is_trading():
        time.sleep(1)  # Refresh every second
        st.rerun()
    
    # Get current data from service
    stats = service.get_stats()
    active_trades = service.get_active_trades()
    closed_trades = service.get_closed_trades()
    
    # Page 1: Active Trades + Live Stats
    if page == "Active Trades":
        
        # Display metrics
        st.header("📊 Live Statistics")
        
        # First row of metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            win_rate = stats.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            wins = stats.get('wins', 0)
            st.metric("Wins", wins)
        
        with col3:
            losses = stats.get('losses', 0)
            st.metric("Losses", losses)
        
        with col4:
            balance = stats.get('balance', 10000)
            st.metric("Balance", f"${balance:,.2f}")
        
        with col5:
            total_pnl = stats.get('total_pnl', 0)
            st.metric(
                "Total P&L (USD)", 
                f"${total_pnl:,.2f}",
                delta=f"{total_pnl:+.2f}"
            )
        
        # Second row of metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_pnl_percent = stats.get('total_pnl_percent', 0)
            st.metric(
                "Total P&L (%)", 
                f"{total_pnl_percent:.2f}%",
                delta=f"{total_pnl_percent:+.2f}%"
            )
        
        with col2:
            margin_used = stats.get('margin_used', 0)
            st.metric("Margin Used", f"${margin_used:,.2f}")
        
        with col3:
            max_drawdown = stats.get('max_drawdown', 0)
            st.metric("Max Drawdown", f"${max_drawdown:,.2f}")
        
        with col4:
            active_count = stats.get('active_positions', 0)
            max_positions = stats.get('max_positions', 10)
            st.metric("Open Trades", f"{active_count}/{max_positions}")
        
        with col5:
            equity = stats.get('equity', 10000)
            st.metric("Equity", f"${equity:,.2f}")
        
        # Active trades table
        st.divider()
        st.header("🔥 Active Trades")
        
        if active_trades:
            # Convert to DataFrame
            df_active = pd.DataFrame(active_trades)
            
            # Select and rename columns for display
            display_columns = {
                'symbol': 'Symbol',
                'strategy': 'Strategy',
                'entry_price': 'Entry Price',
                'current_price': 'Current Price',
                'side': 'Side',
                'leverage': 'Leverage',
                'pnl': 'P&L (USD)',
                'pnl_percent': 'P&L (%)',
                'margin_used': 'Margin Used',
                'position_size_usd': 'Position (USD)',
                'duration': 'Duration (min)',
                'stop_loss': 'Stop Loss',
                'max_pnl_percent': 'Max P&L (%)'
            }
            
            # Filter columns that exist
            existing_cols = [col for col in display_columns.keys() if col in df_active.columns]
            df_display = df_active[existing_cols].copy()
            
            # Rename columns
            df_display.columns = [display_columns[col] for col in existing_cols]
            
            # Format numeric columns
            if 'Entry Price' in df_display.columns:
                df_display['Entry Price'] = df_display['Entry Price'].apply(lambda x: f"${x:.4f}")
            if 'Current Price' in df_display.columns:
                df_display['Current Price'] = df_display['Current Price'].apply(lambda x: f"${x:.4f}")
            if 'P&L (USD)' in df_display.columns:
                df_display['P&L (USD)'] = df_display['P&L (USD)'].apply(lambda x: f"${x:+.2f}")
            if 'P&L (%)' in df_display.columns:
                df_display['P&L (%)'] = df_display['P&L (%)'].apply(lambda x: f"{x:+.2f}%")
            if 'Margin Used' in df_display.columns:
                df_display['Margin Used'] = df_display['Margin Used'].apply(lambda x: f"${x:.2f}")
            if 'Position (USD)' in df_display.columns:
                df_display['Position (USD)'] = df_display['Position (USD)'].apply(lambda x: f"${x:.2f}")
            if 'Duration (min)' in df_display.columns:
                df_display['Duration (min)'] = df_display['Duration (min)'].apply(lambda x: f"{x:.1f}")
            if 'Stop Loss' in df_display.columns:
                df_display['Stop Loss'] = df_display['Stop Loss'].apply(lambda x: f"${x:.4f}")
            if 'Max P&L (%)' in df_display.columns:
                df_display['Max P&L (%)'] = df_display['Max P&L (%)'].apply(lambda x: f"{x:.2f}%")
            if 'Leverage' in df_display.columns:
                df_display['Leverage'] = df_display['Leverage'].apply(lambda x: f"{x}x")
            
            # Display table
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No active trades")
    
    # Page 2: Closed Trades
    elif page == "Closed Trades":
        
        st.header("📜 Closed Trades History")
        
        if closed_trades:
            # Convert to DataFrame
            df_closed = pd.DataFrame(closed_trades)
            
            # Select and rename columns for display
            display_columns = {
                'symbol': 'Symbol',
                'strategy': 'Strategy',
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'side': 'Side',
                'leverage': 'Leverage',
                'pnl': 'P&L (USD)',
                'pnl_percent': 'P&L (%)',
                'duration': 'Duration (min)',
                'close_reason': 'Close Reason',
                'position_size_usd': 'Position (USD)',
                'margin_used': 'Margin Used'
            }
            
            # Filter columns that exist
            existing_cols = [col for col in display_columns.keys() if col in df_closed.columns]
            df_display = df_closed[existing_cols].copy()
            
            # Rename columns
            df_display.columns = [display_columns[col] for col in existing_cols]
            
            # Format numeric columns
            if 'Entry Price' in df_display.columns:
                df_display['Entry Price'] = df_display['Entry Price'].apply(lambda x: f"${x:.4f}")
            if 'Exit Price' in df_display.columns:
                df_display['Exit Price'] = df_display['Exit Price'].apply(lambda x: f"${x:.4f}" if x else "N/A")
            if 'P&L (USD)' in df_display.columns:
                df_display['P&L (USD)'] = df_display['P&L (USD)'].apply(lambda x: f"${x:+.2f}")
            if 'P&L (%)' in df_display.columns:
                df_display['P&L (%)'] = df_display['P&L (%)'].apply(lambda x: f"{x:+.2f}%")
            if 'Duration (min)' in df_display.columns:
                df_display['Duration (min)'] = df_display['Duration (min)'].apply(lambda x: f"{x:.1f}")
            if 'Position (USD)' in df_display.columns:
                df_display['Position (USD)'] = df_display['Position (USD)'].apply(lambda x: f"${x:.2f}")
            if 'Margin Used' in df_display.columns:
                df_display['Margin Used'] = df_display['Margin Used'].apply(lambda x: f"${x:.2f}")
            if 'Leverage' in df_display.columns:
                df_display['Leverage'] = df_display['Leverage'].apply(lambda x: f"{x}x" if x else "N/A")
            
            # Display table
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True
            )
            
            # Summary stats
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            total_closed_pnl = sum(t.get('pnl', 0) for t in closed_trades)
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
            
            with col1:
                st.metric("Total Closed P&L", f"${total_closed_pnl:+,.2f}")
            
            with col2:
                avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
                st.metric("Avg Win", f"${avg_win:,.2f}")
            
            with col3:
                avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
                st.metric("Avg Loss", f"${avg_loss:,.2f}")
            
            with col4:
                st.metric("Total Trades", len(closed_trades))
        else:
            st.info("No closed trades yet")


if __name__ == "__main__":
    main()
