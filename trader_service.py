"""
TraderJoes Headless Trading Service
Wrapper for headless trading worker - NO CODE GENERATION
ALL TRADING LOGIC PRESERVED FROM ORIGINAL
"""

import sys
import json
import asyncio
import logging
import traceback
import threading
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

# Import the headless worker - NO CODE GENERATION
from trading_worker_headless import TradingWorkerV2Headless

logger = logging.getLogger(__name__)


class TraderJoesService:
    """
    Service wrapper for headless trading - NO LOGIC CHANGES
    Only provides thread-safe access to trading data
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the trading service"""
        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            # Default config
            self.config = {
                "api": {
                    "key": "",
                    "secret": "",
                    "testnet": True
                },
                "trading": {
                    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
                    "risk_per_trade": 0.01,
                    "max_positions": 10,
                    "min_confidence": 0.50
                },
                "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
            }
        
        # Ensure symbols are at top level for compatibility
        if 'symbols' not in self.config and 'trading' in self.config:
            self.config['symbols'] = self.config['trading'].get('symbols', ["BTCUSDT"])
        
        # Thread-safe storage (these store EXACT data structures from original)
        self.lock = threading.Lock()
        self.active_trades_view = []   # List of dicts with EXACT structure
        self.closed_trades_view = []   # List of dicts with EXACT structure  
        self.stats_view = {}           # Dict with EXACT structure
        self.log_messages = deque(maxlen=500)
        
        # Trading state
        self._is_running = False
        self.thread = None
        
        # The headless worker instance
        self.worker: Optional[TradingWorkerV2Headless] = None
        
    def start(self):
        """Start trading in background thread"""
        if self._is_running:
            return
            
        self._is_running = True
        self.thread = threading.Thread(target=self._run_trading, daemon=True)
        self.thread.start()
        self.add_log_message("🚀 Trading service started")
        
    def stop(self):
        """Stop trading"""
        self._is_running = False
        if self.worker:
            self.worker.stop()
        self.add_log_message("🛑 Trading service stopped")
        
    def _run_trading(self):
        """Run the trading logic in thread"""
        try:
            # Create headless worker with config and service reference
            self.worker = TradingWorkerV2Headless(self.config, self)
            
            # Run the EXACT trading loop from original
            asyncio.run(self.worker.trade_loop())
            
        except Exception as e:
            self.add_log_message(f"❌ Trading error: {str(e)}")
            logger.error(f"Trading thread error: {e}\n{traceback.format_exc()}")
            
    def add_log_message(self, message: str):
        """Thread-safe logging - called by worker"""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            full_message = f"[{timestamp}] {message}"
            self.log_messages.append(full_message)
            print(full_message)  # Also print to console
            
    def update_stats(self, stats: dict):
        """Update stats view - called by worker with EXACT structure"""
        with self.lock:
            self.stats_view = stats
            
    def update_active_trades(self, trades: list):
        """Update active trades view - called by worker with EXACT structure"""
        with self.lock:
            self.active_trades_view = trades
            
    def update_closed_trades(self, trades: list):
        """Update closed trades view - called by worker with EXACT structure"""
        with self.lock:
            self.closed_trades_view = trades
            
    # Public API for Streamlit - NO LOGIC, just returns stored data
    def is_trading(self) -> bool:
        """Check if currently trading"""
        return self._is_running
        
    def get_stats(self) -> dict:
        """Get current stats - EXACT structure from original"""
        with self.lock:
            return dict(self.stats_view)
            
    def get_active_trades(self) -> list:
        """Get active trades - EXACT structure from original"""
        with self.lock:
            return list(self.active_trades_view)
            
    def get_closed_trades(self) -> list:
        """Get closed trades - EXACT structure from original"""
        with self.lock:
            return list(self.closed_trades_view)
            
    def get_logs(self, limit: int = 200) -> list:
        """Get recent logs"""
        with self.lock:
            logs = list(self.log_messages)
            return logs[-limit:] if len(logs) > limit else logs
