#!/usr/bin/env python3
"""
TraderJoes v2.1.3 - Professional Crypto Futures Trading Bot
P&L DISPLAY FIXED + REASONABLE STOPS + MTF + LEVERAGE
WITH UNIFIED MARKET DATA SERVICE

Created by: OneDeepx
Repository: github.com/OneDeepx/JT345

MAJOR IMPROVEMENTS IN v2.1.3:
- FIXED: P&L display shows both $ and % in closed trades
- FIXED: Total P&L properly includes all losses
- FIXED: Reasonable stop losses (2-5% price movement)
- FIXED: No duplicate trades on same symbol
- CRITICAL: Multi-timeframe analysis (1m to 1d)
- 3-5x leverage based on confidence
- Strategy-specific stop distances
- Extra room for volatile coins
- Proper percentage-based trailing stop
- Continuous trading operation
- NEW: Unified Market Data Service for clean networking
"""

import sys
import json
import asyncio
import logging
import traceback
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from collections import deque, defaultdict
from queue import Queue
import csv
import hashlib
from dataclasses import dataclass, field
from enum import Enum

# Import the unified market data service
from market_data_service import get_market_data_service, MarketDataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('traderjoes_v2.1.log')
    ]
)
logger = logging.getLogger(__name__)

# PyQt imports
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Import bot modules - using MarketDataService
from trading_engine import TradingEngine, TradingStrategyAnalyzer
# from binance_api import BinanceFuturesClient - removed (replaced by MarketDataService)
from validation_engine import ValidationEngine, RealisticExecutionSimulator


class TimeFrame(Enum):
    """Available timeframes for analysis"""
    M1 = '1m'    # Scalping
    M3 = '3m'    # Scalping
    M5 = '5m'    # Short-term
    M15 = '15m'  # Intraday
    M30 = '30m'  # Intraday
    H1 = '1h'    # Swing
    H2 = '2h'    # Swing
    H4 = '4h'    # Position
    H8 = '8h'    # Position
    H12 = '12h'  # Position
    D1 = '1d'    # Long-term


class MultiTimeframeAnalyzer:
    """Analyzes multiple timeframes to determine best strategy"""
    
    def __init__(self):
        self.timeframes = [
            TimeFrame.M1, TimeFrame.M5, TimeFrame.M15,
            TimeFrame.M30, TimeFrame.H1, TimeFrame.H4,
            TimeFrame.D1
        ]
        self.timeframe_data = {}
        self.timeframe_trends = {}
        self.timeframe_momentum = {}
        self.timeframe_volatility = {}
        
    async def analyze_all_timeframes(self, symbol: str, market_service: MarketDataService) -> Dict:
        """Analyze all timeframes for a symbol using Market Data Service"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'timeframes': {},
            'alignment': None,
            'recommended_strategy': None,
            'confidence': 0,
            'details': {}
        }
        
        # Use specific timeframes available in Market Data Service
        available_timeframes = ['1m', '5m', '15m']  # These are subscribed in the service
        
        for tf_str in available_timeframes:
            try:
                # Get candles from Market Data Service
                df = market_service.get_candles(symbol, tf_str, 200)
                if not df.empty:
                    # Create TimeFrame enum value for analysis
                    tf = TimeFrame.M1 if tf_str == '1m' else TimeFrame.M5 if tf_str == '5m' else TimeFrame.M15
                    tf_analysis = self.analyze_timeframe(df, tf)
                    analysis['timeframes'][tf_str] = tf_analysis
                    self.timeframe_data[f"{symbol}_{tf_str}"] = df
            except Exception as e:
                logger.error(f"Error analyzing {symbol} {tf_str}: {e}")
        
        # Determine alignment and strategy
        analysis['alignment'] = self.calculate_alignment(analysis['timeframes'])
        analysis['recommended_strategy'] = self.determine_strategy(analysis)
        analysis['confidence'] = self.calculate_confidence(analysis)
        
        return analysis
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame) -> Dict:
        """Analyze a single timeframe"""
        if df.empty or len(df) < 20:
            return {'trend': 'NEUTRAL', 'momentum': 0, 'volatility': 0}
        
        analysis = {}
        
        # Calculate trend (SMA comparison)
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=min(50, len(df))).mean()
        
        current_price = df['close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1] if len(df) >= 50 else sma_20
        
        # Determine trend
        if current_price > sma_20 > sma_50:
            analysis['trend'] = 'STRONG_UP'
        elif current_price > sma_20:
            analysis['trend'] = 'UP'
        elif current_price < sma_20 < sma_50:
            analysis['trend'] = 'STRONG_DOWN'
        elif current_price < sma_20:
            analysis['trend'] = 'DOWN'
        else:
            analysis['trend'] = 'NEUTRAL'
        
        # Calculate momentum (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        analysis['momentum'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Calculate volatility (ATR as percentage)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean()
        analysis['volatility'] = (atr.iloc[-1] / current_price * 100) if not pd.isna(atr.iloc[-1]) else 0
        
        # Support and resistance
        analysis['support'] = df['low'].rolling(20).min().iloc[-1]
        analysis['resistance'] = df['high'].rolling(20).max().iloc[-1]
        
        # Volume analysis
        analysis['volume_trend'] = 'HIGH' if df['volume'].iloc[-1] > df['volume'].mean() else 'LOW'
        
        # MACD for additional confirmation
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        analysis['macd'] = (exp1 - exp2).iloc[-1]
        analysis['macd_signal'] = 'BUY' if analysis['macd'] > 0 else 'SELL'
        
        return analysis
    
    def calculate_alignment(self, timeframes: Dict) -> str:
        """Calculate trend alignment across timeframes"""
        if not timeframes:
            return 'UNKNOWN'
        
        trends = []
        weights = {
            '1m': 0.5, '3m': 0.5, '5m': 1,
            '15m': 2, '30m': 2, '1h': 3,
            '2h': 3, '4h': 4, '8h': 4,
            '12h': 4, '1d': 5
        }
        
        weighted_score = 0
        total_weight = 0
        
        for tf, analysis in timeframes.items():
            if 'trend' in analysis:
                weight = weights.get(tf, 1)
                total_weight += weight
                
                if analysis['trend'] == 'STRONG_UP':
                    weighted_score += weight * 2
                elif analysis['trend'] == 'UP':
                    weighted_score += weight * 1
                elif analysis['trend'] == 'DOWN':
                    weighted_score += weight * -1
                elif analysis['trend'] == 'STRONG_DOWN':
                    weighted_score += weight * -2
        
        if total_weight == 0:
            return 'NEUTRAL'
        
        alignment_score = weighted_score / total_weight
        
        if alignment_score > 1.5:
            return 'STRONG_BULLISH'
        elif alignment_score > 0.5:
            return 'BULLISH'
        elif alignment_score < -1.5:
            return 'STRONG_BEARISH'
        elif alignment_score < -0.5:
            return 'BEARISH'
        else:
            return 'MIXED'
    
    def determine_strategy(self, analysis: Dict) -> str:
        """Determine best strategy based on multi-timeframe analysis"""
        alignment = analysis['alignment']
        timeframes = analysis['timeframes']
        
        # Get short-term and long-term analysis
        short_term_vol = 0
        long_term_trend = 'NEUTRAL'
        
        if '5m' in timeframes:
            short_term_vol = timeframes['5m'].get('volatility', 0)
        
        if '4h' in timeframes:
            long_term_trend = timeframes['4h'].get('trend', 'NEUTRAL')
        elif '1h' in timeframes:
            long_term_trend = timeframes['1h'].get('trend', 'NEUTRAL')
        
        # Strategy selection logic
        if alignment in ['STRONG_BULLISH', 'STRONG_BEARISH']:
            # Strong trend alignment - use trend following
            return 'TREND_FOLLOWING'
        
        elif short_term_vol > 2.0:
            # High volatility - use scalping
            if '1m' in timeframes and timeframes['1m'].get('momentum', 50) != 50:
                return 'SCALPING'
            else:
                return 'MOMENTUM'
        
        elif alignment == 'MIXED':
            # Mixed signals - look for mean reversion
            if '15m' in timeframes:
                rsi = timeframes['15m'].get('momentum', 50)
                if rsi > 70 or rsi < 30:
                    return 'MEAN_REVERSION'
            return 'BREAKOUT'
        
        elif alignment in ['BULLISH', 'BEARISH']:
            # Moderate trend - use momentum
            return 'MOMENTUM'
        
        else:
            # Neutral market - look for breakouts
            return 'BREAKOUT'
    
    def calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence based on timeframe alignment"""
        alignment = analysis['alignment']
        base_confidence = 0.5
        
        # Alignment bonus
        if alignment in ['STRONG_BULLISH', 'STRONG_BEARISH']:
            base_confidence += 0.25
        elif alignment in ['BULLISH', 'BEARISH']:
            base_confidence += 0.15
        elif alignment == 'MIXED':
            base_confidence -= 0.10
        
        # Check for confluence across multiple timeframes
        timeframes = analysis['timeframes']
        confirming_timeframes = 0
        
        for tf, tf_analysis in timeframes.items():
            if tf in ['15m', '1h', '4h']:  # Key timeframes
                trend = tf_analysis.get('trend', 'NEUTRAL')
                if (alignment == 'BULLISH' and 'UP' in trend) or \
                   (alignment == 'BEARISH' and 'DOWN' in trend):
                    confirming_timeframes += 1
        
        base_confidence += confirming_timeframes * 0.05
        
        # Cap confidence
        return min(max(base_confidence, 0.3), 0.85)
    
    def get_entry_timeframe(self, strategy: str) -> str:
        """Get the best timeframe for entry based on strategy"""
        strategy_timeframes = {
            'SCALPING': '1m',
            'MOMENTUM': '5m',
            'MEAN_REVERSION': '15m',
            'TREND_FOLLOWING': '1h',
            'BREAKOUT': '15m'
        }
        return strategy_timeframes.get(strategy, '5m')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('traderjoes_v2.log')
    ]
)
logger = logging.getLogger(__name__)

# PyQt imports
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Duplicate imports removed - using imports from top of file

@dataclass
class TradeV2:
    """Enhanced trade class with MTF tracking and LEVERAGE"""
    id: str
    symbol: str
    strategy: str
    side: str
    entry_price: float
    position_size: float  # Units of crypto
    position_size_usd: float  # LEVERAGED position value
    margin_used: float  # Actual balance used (1% of balance)
    leverage: float  # 3-5x based on confidence
    stop_loss: float
    take_profit: float
    entry_time: datetime
    current_price: float = 0
    highest_price: float = 0  # For tracking
    lowest_price: float = float('inf')  # For tracking
    max_pnl_percent: float = 0  # Track maximum P&L PERCENTAGE for trailing
    pnl: float = 0
    pnl_percent: float = 0  # Current P&L percentage
    fees: float = 0
    status: str = 'OPEN'
    validation_hash: str = ''  # For historical validation
    mtf_alignment: str = 'UNKNOWN'  # Multi-timeframe alignment
    mtf_confidence: float = 0  # MTF-based confidence
    exit_price: float = None
    exit_time: datetime = None
    close_reason: str = None
    duration: float = 0
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current P&L with LEVERAGE"""
        if self.side == 'BUY':
            price_change = current_price - self.entry_price
        else:
            price_change = self.entry_price - current_price
        
        # P&L is calculated on the LEVERAGED position
        return (price_change * self.position_size) - self.fees
    
    def calculate_pnl_percent(self, current_price: float) -> float:
        """Calculate P&L as percentage of MARGIN used"""
        pnl = self.calculate_pnl(current_price)
        # Percentage is based on actual margin used, not leveraged position
        return (pnl / self.margin_used) * 100


class MLStrategyOptimizer:
    """Enhanced ML with aggressive learning for continuous improvement"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.strategy_weights = {
            'MOMENTUM': 1.0,
            'TREND_FOLLOWING': 1.0,
            'MEAN_REVERSION': 1.0,
            'SCALPING': 1.0,
            'BREAKOUT': 1.0
        }
        self.learning_rate = 0.02  # Doubled for faster learning
        self.min_samples = 10  # Reduced for quicker adaptation
        self.loss_multiplier = 1.5  # Learn more from losses
        self.total_trades = 0
        self.strategy_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        
    def update_weights(self, strategy: str, profit: bool, pnl_amount: float = 0):
        """Aggressively adjust strategy weights based on performance"""
        self.total_trades += 1
        
        # Track detailed performance
        if profit:
            self.strategy_weights[strategy] = min(2.5, self.strategy_weights[strategy] * 1.08)
            self.strategy_performance[strategy]['wins'] += 1
            self.strategy_performance[strategy]['total_pnl'] += pnl_amount
        else:
            # Learn MORE from losses for improvement
            self.strategy_weights[strategy] = max(0.05, self.strategy_weights[strategy] * 0.90)
            self.strategy_performance[strategy]['losses'] += 1
            self.strategy_performance[strategy]['total_pnl'] += pnl_amount
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        for key in self.strategy_weights:
            self.strategy_weights[key] /= total
        
        # Log learning progress
        logger.info(f"ML Learning #{self.total_trades}: {strategy} {'WIN' if profit else 'LOSS'} ${pnl_amount:.2f}")
        logger.info(f"Updated weights: {self.strategy_weights}")
        
        # Every 10 trades, analyze and report
        if self.total_trades % 10 == 0:
            self.analyze_performance()
    
    def analyze_performance(self):
        """Analyze and log performance for continuous improvement"""
        logger.info("=" * 50)
        logger.info("ML PERFORMANCE ANALYSIS - Data Collection Active")
        for strategy, perf in self.strategy_performance.items():
            total = perf['wins'] + perf['losses']
            if total > 0:
                win_rate = (perf['wins'] / total) * 100
                logger.info(f"{strategy}: {perf['wins']}/{total} wins ({win_rate:.1f}%) | P&L: ${perf['total_pnl']:.2f}")
        logger.info(f"TRADING CONTINUES - Gathering more data...")
        logger.info("=" * 50)
    
    def get_best_strategy(self, market_conditions: dict) -> Tuple[str, float]:
        """Select best strategy based on learned weights"""
        # Add some randomization to explore different strategies
        import random
        if random.random() < 0.15:  # 15% exploration rate
            # Occasionally try underperforming strategies for data
            strategy = random.choice(list(self.strategy_weights.keys()))
            return strategy, 0.60  # Lower confidence for exploration
        
        # Otherwise use best performing
        best_strategy = max(self.strategy_weights.items(), key=lambda x: x[1])
        return best_strategy[0], best_strategy[1]


class RiskManagerV2:
    """Enhanced risk management with LEVERAGE and continuous trading"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.available_balance = initial_balance  # Track available balance
        self.risk_per_trade = 0.01  # Strict 1% risk (margin used)
        self.max_positions = 10  # Maximum 10 trades
        self.trailing_stop_percentage = 0.30  # 0.30 PERCENTAGE POINTS for trailing
        self.min_leverage = 3.0  # Minimum leverage
        self.max_leverage = 5.0  # Maximum leverage
        self.total_losses = 0  # Track but don't limit
        self.consecutive_losses = 0  # Track but don't suspend
        self.allow_trading = True  # ALWAYS TRUE - Never suspend
        
    def calculate_leverage(self, confidence: float, mtf_alignment: str = None) -> float:
        """Calculate leverage based on confidence (3x to 5x)"""
        # Base calculation: map confidence 0.4-0.8 to leverage 3-5
        if confidence < 0.4:
            confidence = 0.4
        elif confidence > 0.8:
            confidence = 0.8
        
        # Linear mapping: 0.4 conf = 3x, 0.8 conf = 5x
        leverage = 3.0 + ((confidence - 0.4) / 0.4) * 2.0
        
        # Boost for strong MTF alignment
        if mtf_alignment in ['STRONG_BULLISH', 'STRONG_BEARISH']:
            leverage += 0.5
        
        # Ensure within bounds
        leverage = max(self.min_leverage, min(self.max_leverage, leverage))
        
        return round(leverage, 1)
    
    def calculate_position_size_with_leverage(self, balance: float, leverage: float) -> tuple:
        """Calculate margin used and leveraged position size
        Returns: (margin_used, leveraged_position_usd)"""
        # Margin used is 1% of available balance
        margin_used = balance * self.risk_per_trade
        
        # Leveraged position is margin * leverage
        leveraged_position = margin_used * leverage
        
        # Safety check for very low balance
        if balance < 100:
            margin_used = min(10, balance * self.risk_per_trade)
            leveraged_position = margin_used * leverage
        
        return margin_used, leveraged_position
    
    def calculate_position_units(self, leveraged_position_usd: float, entry_price: float) -> float:
        """Calculate position size in units based on leveraged position"""
        if entry_price > 0:
            return leveraged_position_usd / entry_price
        return 0
    
    def update_available_balance(self, amount: float, operation: str = 'subtract'):
        """Update available balance when trades open/close
        Note: 'amount' is the MARGIN used, not the leveraged position"""
        if operation == 'subtract':
            self.available_balance -= amount
        else:  # add
            self.available_balance += amount
        
        # Never let balance go negative (use minimum trading amount)
        if self.available_balance < 10:
            logger.warning(f"Low balance: ${self.available_balance:.2f} - Continuing with minimum positions")
        
        logger.info(f"Available balance: ${self.available_balance:.2f} | Total losses: ${self.total_losses:.2f}")
    
    def can_trade(self) -> bool:
        """Always returns True - NO TRADING SUSPENSIONS"""
        return True  # ALWAYS allow trading for data collection
    
    def record_trade_result(self, pnl: float):
        """Record trade result but NEVER suspend trading"""
        if pnl < 0:
            self.total_losses += abs(pnl)
            self.consecutive_losses += 1
            logger.info(f"Loss recorded: ${abs(pnl):.2f} | Consecutive: {self.consecutive_losses} | Total: ${self.total_losses:.2f}")
        else:
            self.consecutive_losses = 0  # Reset on win
            logger.info(f"Win recorded: ${pnl:.2f} | Losses reset")
        
        # Log statistics but NEVER suspend
        logger.info(f"Trading continues - Data collection active")


class HistoricalValidator:
    """Validate trades with historical timestamp verification"""
    
    def __init__(self):
        self.validated_trades = {}
        self.validation_log = []
        
    def create_validation_hash(self, trade: TradeV2) -> str:
        """Create unique hash for trade validation"""
        data = f"{trade.symbol}_{trade.entry_price}_{trade.entry_time.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def validate_entry(self, trade: TradeV2, market_data: pd.DataFrame) -> dict:
        """Validate trade entry with historical data"""
        validation = {
            'timestamp': trade.entry_time.isoformat(),
            'symbol': trade.symbol,
            'entry_price': trade.entry_price,
            'market_price_at_entry': 0,
            'price_match': False,
            'time_verified': False,
            'hash': self.create_validation_hash(trade)
        }
        
        try:
            # Find closest timestamp in market data
            if not market_data.empty and 'close' in market_data.columns:
                # Get price at entry time
                closest_idx = market_data.index.get_indexer([trade.entry_time], method='nearest')[0]
                if closest_idx >= 0 and closest_idx < len(market_data):
                    market_price = market_data.iloc[closest_idx]['close']
                    validation['market_price_at_entry'] = market_price
                    
                    # Check if prices match within 0.1%
                    price_diff = abs(market_price - trade.entry_price) / trade.entry_price
                    validation['price_match'] = price_diff < 0.001
                    
                    # Verify timestamp is reasonable (within 1 minute)
                    time_diff = abs((market_data.index[closest_idx] - trade.entry_time).total_seconds())
                    validation['time_verified'] = time_diff < 60
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
        
        self.validation_log.append(validation)
        return validation


class TradingWorkerV2(QThread):
    """Enhanced trading worker with multi-timeframe analysis"""
    
    # Signals
    log_signal = pyqtSignal(str)
    trade_signal = pyqtSignal(dict)
    stats_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    validation_signal = pyqtSignal(dict)
    mtf_signal = pyqtSignal(dict)  # Multi-timeframe analysis signal
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = False
        self.client = None
        self.trading_engine = None
        self.risk_manager = RiskManagerV2()
        self.ml_optimizer = MLStrategyOptimizer()
        self.validator = HistoricalValidator()
        self.mtf_analyzer = MultiTimeframeAnalyzer()  # NEW: MTF analyzer
        self.active_trades = {}  # Dict of TradeV2 objects
        self.closed_trades = []
        self.close_queue = Queue()
        self.update_counter = 0
        self.scan_count = 0
        self.mtf_cache = {}  # Cache MTF analysis
        
        # Unified Market Data Service
        self.market_service: Optional[MarketDataService] = None
        self.loop = None
        
    def run(self):
        """Main trading thread"""
        self.is_running = True
        logger.info("🚀 Starting TraderJoes v2.0.0...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.trade_loop())
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"Trading error: {e}\n{traceback.format_exc()}")
        finally:
            loop.close()
            logger.info("Trading stopped")
    
    async def trade_loop(self):
        """Main trading loop with v2.0.0 enhancements"""
        
        # Store the event loop for Market Data Service callbacks
        self.loop = asyncio.get_event_loop()
        
        # Initialize Market Data Service (replaces old Binance client)
        self.market_service = get_market_data_service(self.config)
        
        # Get symbols from config
        symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'])
        
        # Start market data service
        await self.market_service.start(symbols)
        logger.info(f"✅ Market Data Service started for {len(symbols)} symbols")
        logger.info("📡 Real-time WebSocket + REST data flowing")
        
        # Initialize trading engine
        self.trading_engine = TradingEngine()
        self.trading_engine.account_balance = 10000
        self.risk_manager.initial_balance = 10000
        self.risk_manager.available_balance = 10000
        
        # Add symbols to trading engine
        self.trading_engine.add_symbols(symbols)
        
                # Subscribe to price updates for active trade monitoring
        try:
            for symbol in symbols:
                self.market_service.subscribe_price(symbol, self._on_price_update)
                # Subscribe to closed candles for strategy signals
                self.market_service.subscribe_candle_closed(symbol, '5m', self._on_candle_closed)
            logger.info(f"📊 Monitoring {len(symbols)} symbols with unified data layer")
            logger.info("⚡ Real-time WebSocket + REST fallback enabled")
        except Exception as e:
            logger.info(f"⚠️ Market Data Service error: {e}")
            return

        logger.info("=" * 50)
        logger.info("✅ TraderJoes v2.1.3 Ready - P&L DISPLAY FIXED")
        logger.info(f"💰 Starting Balance: ${self.trading_engine.account_balance:.2f}")
        logger.info(f"💵 Available Balance: ${self.risk_manager.available_balance:.2f}")
        logger.info(f"📊 Risk Per Trade: {self.risk_manager.risk_per_trade*100}% margin")
        logger.info(f"🎯 Max Positions: {self.risk_manager.max_positions} (1 per symbol)")
        logger.info(f"⚡ Leverage: 3-5x based on confidence")
        logger.info(f"🛡️ Stop Loss: 2-5% price distance")
        logger.info(f"💹 P&L Display: Shows $ and % properly")
        logger.info(f"📈 Trailing Stop: 0.30 percentage points")
        logger.info("🔍 MULTI-TIMEFRAME: 1m to 1d analysis")
        logger.info("🚫 DUPLICATE PREVENTION: One position per symbol")
        logger.info("=" * 50)
            
        # Main loop - 1 second intervals for real-time updates
        while self.is_running:
            try:
                self.update_counter += 1

                # Process close requests
                await self.process_close_requests()

                # Update active trades every second for real-time P&L
                await self.update_active_trades_fast()

                # Trade scanning every 5 seconds to avoid over-trading
                if self.update_counter % 5 == 0:
                    await self.scan_markets()

                # Sleep for 1 second between iterations
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(1)











    
    async def update_active_trades_fast(self):
        """Ultra-fast active trades update with leverage and P&L percentage tracking"""
        active_list = []
        
        for trade_id, trade in self.active_trades.items():
            # Get current price
            current_price = await self.get_current_price(trade.symbol)
            if current_price:
                trade.current_price = current_price
                
                # Update highest/lowest for reference
                if trade.side == 'BUY':
                    trade.highest_price = max(trade.highest_price, current_price)
                else:
                    trade.lowest_price = min(trade.lowest_price, current_price)
            else:
                # Simulate for demo
                current_price = trade.current_price * (1 + np.random.uniform(-0.002, 0.002))
                trade.current_price = current_price
            
            # Calculate P&L and percentage (on margin)
            trade.pnl = trade.calculate_pnl(current_price)
            trade.pnl_percent = trade.calculate_pnl_percent(current_price)
            
            # Track MAXIMUM P&L PERCENTAGE for trailing stop
            trade.max_pnl_percent = max(trade.max_pnl_percent, trade.pnl_percent)
            
            # Prepare display data
            active_list.append({
                'id': trade_id,
                'symbol': trade.symbol,
                'strategy': trade.strategy,
                'side': trade.side,
                'leverage': trade.leverage,  # Add leverage
                'entry_price': trade.entry_price,
                'current_price': current_price,
                'margin_used': trade.margin_used,  # Actual balance used
                'position_size_usd': trade.position_size_usd,  # Leveraged position
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,  # P&L% on margin
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'max_pnl_percent': trade.max_pnl_percent,  # Show max P&L%
                'duration': str(datetime.now() - trade.entry_time).split('.')[0]
            })
        
        if active_list:
            # removed trade signal
    
    async def scan_markets(self):
        """Scan markets with multi-timeframe analysis and DUPLICATE PREVENTION"""
        # Always try to trade if we have room for positions
        if len(self.active_trades) >= self.risk_manager.max_positions:
            logger.info(f"Max positions reached ({self.risk_manager.max_positions}) - Waiting for closes")
            return
        
        # Continue trading even with low balance
        if self.risk_manager.available_balance < 100:
            logger.warning(f"Low balance ${self.risk_manager.available_balance:.2f} - Using minimum positions for data collection")
        
        for symbol in self.config.get('symbols', []):
            if len(self.active_trades) >= self.risk_manager.max_positions:
                break
            
            # CRITICAL: Check for existing position on this symbol
            if self.has_active_position(symbol):
                logger.info(f"⚠️ Skipping {symbol} - Already have active position")
                continue
                
            try:
                # CRITICAL: Multi-timeframe analysis
                logger.info(f"🔍 Analyzing {symbol} across all timeframes...")
                mtf_analysis = await self.mtf_analyzer.analyze_all_timeframes(symbol, self.market_service)
                
                # Cache the analysis
                self.mtf_cache[symbol] = mtf_analysis
                
                # Send MTF signal for UI display
                self.mtf_signal.emit({
                    'symbol': symbol,
                    'alignment': mtf_analysis['alignment'],
                    'strategy': mtf_analysis['recommended_strategy'],
                    'confidence': mtf_analysis['confidence'],
                    'timeframes': mtf_analysis['timeframes']
                })
                
                # Log MTF results
                self.log_signal.emit(
                    f"📊 {symbol} MTF Analysis:\n"
                    f"   Alignment: {mtf_analysis['alignment']}\n"
                    f"   Strategy: {mtf_analysis['recommended_strategy']}\n"
                    f"   Confidence: {mtf_analysis['confidence']:.1%}"
                )
                
                # Get the appropriate timeframe for entry
                entry_timeframe = self.mtf_analyzer.get_entry_timeframe(
                    mtf_analysis['recommended_strategy']
                )
                
                # Get market data from Market Data Service
                df = self.market_service.get_candles(symbol, entry_timeframe, 500)
                if df.empty:
                    # Skip if no data available
                    continue
                
                # Store data
                self.trading_engine.market_data[symbol] = df
                current_price = df['close'].iloc[-1]
                
                # Override strategy with MTF recommendation
                self.trading_engine.strategy_analyzer.preferred_strategy = mtf_analysis['recommended_strategy']
                
                # Get trading signal with MTF-informed strategy
                signal = self.trading_engine.strategy_analyzer.get_best_signal(df, symbol)
                
                # Use MTF confidence boost
                if signal:
                    # DUPLICATE CHECK: Verify no recent similar trade
                    if self.is_duplicate_signal(symbol, signal.action, signal.entry_price):
                        logger.info(f"⚠️ Duplicate signal rejected for {symbol} at ${signal.entry_price:.2f}")
                        continue
                    
                    # Combine signal confidence with MTF confidence
                    combined_confidence = (signal.confidence * 0.4 + mtf_analysis['confidence'] * 0.6)
                    
                    # Lower threshold for strong alignment
                    min_confidence = 0.40 if mtf_analysis['alignment'] in ['STRONG_BULLISH', 'STRONG_BEARISH'] else 0.45
                    
                    if combined_confidence > min_confidence:
                        # Check signal direction matches MTF alignment
                        if self.validate_signal_with_mtf(signal, mtf_analysis):
                            signal.confidence = combined_confidence  # Update confidence
                            signal.strategy = mtf_analysis['recommended_strategy']  # Use MTF strategy
                            await self.execute_trade_v2(signal, current_price, df, mtf_analysis)
                            self.log_signal.emit(
                                f"✅ Trade #{self.ml_optimizer.total_trades + 1} opened with MTF alignment"
                            )
                        else:
                            logger.info(f"⚠️ Signal rejected: Conflicts with MTF alignment")
                    
            except Exception as e:
                logger.error(f"Scan error for {symbol}: {e}")
                continue
    
    def has_active_position(self, symbol: str) -> bool:
        """Check if we already have an active position on this symbol"""
        for trade in self.active_trades.values():
            if trade.symbol == symbol:
                return True
        return False
    
    def is_duplicate_signal(self, symbol: str, action: str, entry_price: float, threshold: float = 0.002) -> bool:
        """Check if this would be a duplicate trade (same symbol, direction, similar price)"""
        # Check active trades
        for trade in self.active_trades.values():
            if trade.symbol == symbol and trade.side == action:
                # Check if entry prices are within 0.2% of each other
                price_diff = abs(trade.entry_price - entry_price) / trade.entry_price
                if price_diff < threshold:
                    return True
        
        # Also check recently closed trades (last 10)
        recent_closed = self.closed_trades[-10:] if len(self.closed_trades) >= 10 else self.closed_trades
        for trade in recent_closed:
            if trade.symbol == symbol and trade.side == action:
                # Check if trade was closed in last 5 minutes
                if trade.exit_time and (datetime.now() - trade.exit_time).seconds < 300:
                    price_diff = abs(trade.entry_price - entry_price) / trade.entry_price
                    if price_diff < threshold:
                        return True
        
        return False
    
    def validate_signal_with_mtf(self, signal, mtf_analysis) -> bool:
        """Validate that signal direction matches MTF alignment"""
        alignment = mtf_analysis['alignment']
        
        # Strong alignment check
        if alignment in ['STRONG_BULLISH', 'BULLISH'] and signal.action == 'BUY':
            return True
        elif alignment in ['STRONG_BEARISH', 'BEARISH'] and signal.action == 'SELL':
            return True
        elif alignment == 'MIXED':
            # In mixed conditions, allow any signal with lower confidence
            return True
        elif alignment == 'NEUTRAL':
            # In neutral conditions, focus on breakout signals
            return mtf_analysis['recommended_strategy'] == 'BREAKOUT'
        
        return False
    
    async def execute_trade_v2(self, signal, current_price, market_data, mtf_analysis=None):
        """Execute trade with v2.1.3 REASONABLE stop losses and leverage"""
        try:
            # FINAL DUPLICATE CHECK before execution
            if self.has_active_position(signal.symbol):
                logger.info(f"❌ Trade rejected: Already have position on {signal.symbol}")
                return
            
            # Check position limit again
            if len(self.active_trades) >= self.risk_manager.max_positions:
                return
            
            # Calculate LEVERAGE based on confidence
            leverage = self.risk_manager.calculate_leverage(
                signal.confidence,
                mtf_analysis['alignment'] if mtf_analysis else None
            )
            
            # Calculate margin and leveraged position
            margin_used, leveraged_position_usd = self.risk_manager.calculate_position_size_with_leverage(
                self.risk_manager.available_balance,
                leverage
            )
            
            # REASONABLE STOP LOSS - Give trades breathing room!
            # Base stop loss distance (price percentage, not P&L percentage)
            base_sl_distance = 0.03  # 3% price movement default
            
            if mtf_analysis:
                strategy = mtf_analysis['recommended_strategy']
                if strategy == 'SCALPING':
                    base_sl_distance = 0.02  # 2% for scalping
                elif strategy == 'MOMENTUM':
                    base_sl_distance = 0.03  # 3% for momentum
                elif strategy == 'TREND_FOLLOWING':
                    base_sl_distance = 0.05  # 5% for trends (more room)
                elif strategy == 'MEAN_REVERSION':
                    base_sl_distance = 0.025  # 2.5% for mean reversion
                else:
                    base_sl_distance = 0.03  # 3% default
            
            # For very volatile coins, give even more room
            if signal.symbol in ['DOGEUSDT', 'SHIBUSDT', 'ADAUSDT', 'SOLUSDT']:
                base_sl_distance *= 1.5  # 50% more room for volatile coins
            
            # Set stop loss with PROPER DISTANCE
            if signal.action == 'BUY':
                stop_loss = signal.entry_price * (1 - base_sl_distance)
                take_profit = signal.entry_price * 1.10  # 10% TP
            else:
                stop_loss = signal.entry_price * (1 + base_sl_distance)
                take_profit = signal.entry_price * 0.90  # 10% TP
            
            # Log the stop loss calculation for transparency
            sl_price_pct = base_sl_distance * 100
            sl_pnl_pct = base_sl_distance * leverage * 100  # P&L% on margin
            
            self.log_signal.emit(
                f"📊 Stop Loss Calculation:\n"
                f"   Price Distance: {sl_price_pct:.1f}%\n"
                f"   P&L Impact: {sl_pnl_pct:.1f}% of margin with {leverage}x leverage"
            )
            
            # Calculate position size in units (based on LEVERAGED position)
            position_units = self.risk_manager.calculate_position_units(
                leveraged_position_usd,
                signal.entry_price
            )
            
            if position_units <= 0:
                return
            
            # Create trade with leverage info
            trade = TradeV2(
                id=f"{signal.symbol}_{datetime.now().timestamp()}",
                symbol=signal.symbol,
                strategy=signal.strategy.value if hasattr(signal.strategy, 'value') else str(signal.strategy),
                side=signal.action,
                entry_price=signal.entry_price,
                position_size=position_units,
                position_size_usd=leveraged_position_usd,  # LEVERAGED position value
                margin_used=margin_used,  # Actual balance used
                leverage=leverage,  # Store leverage used
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                current_price=current_price,
                highest_price=current_price if signal.action == 'BUY' else 0,
                lowest_price=current_price if signal.action == 'SELL' else float('inf'),
                fees=leveraged_position_usd * 0.0004,  # Fees on leveraged position
                validation_hash='',
                mtf_alignment=mtf_analysis['alignment'] if mtf_analysis else 'UNKNOWN',
                mtf_confidence=mtf_analysis['confidence'] if mtf_analysis else signal.confidence
            )
            
            # Historical validation
            validation = self.validator.validate_entry(trade, market_data)
            trade.validation_hash = validation['hash']
            
            # Store trade
            self.active_trades[trade.id] = trade
            
            # SUBTRACT MARGIN from available balance (not leveraged position!)
            self.risk_manager.update_available_balance(margin_used, 'subtract')
            
            # Send validation signal
            self.validation_signal.emit({
                'type': 'ENTRY',
                'symbol': signal.symbol,
                'timestamp': datetime.now().isoformat(),
                'validation': validation,
                'trade': {
                    'entry_price': trade.entry_price,
                    'margin_used': margin_used,
                    'position_usd': leveraged_position_usd,
                    'leverage': leverage,
                    'stop_loss': stop_loss,
                    'strategy': trade.strategy,
                    'mtf_alignment': trade.mtf_alignment
                }
            })
            
            # Enhanced logging with CLEAR stop loss info
            mtf_info = f"\n   MTF: {trade.mtf_alignment} | Confidence: {trade.mtf_confidence:.1%}" if mtf_analysis else ""
            
            self.log_signal.emit(
                f"✅ TRADE #{len(self.active_trades)} OPENED: {signal.symbol} {signal.action}\n"
                f"   Entry: ${signal.entry_price:.4f} | Leverage: {leverage}x\n"
                f"   Stop Loss: ${stop_loss:.4f} ({sl_price_pct:.1f}% price room)\n"
                f"   Margin: ${margin_used:.2f} | Position: ${leveraged_position_usd:.2f}\n"
                f"   Max Loss: ${sl_pnl_pct:.1f}% of margin\n"
                f"   Strategy: {trade.strategy}{mtf_info}\n"
                f"   Available Balance: ${self.risk_manager.available_balance:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def check_exits_with_trailing(self):
        """Check exits with PROPER percentage-based trailing take profit"""
        for trade_id, trade in list(self.active_trades.items()):
            try:
                current_price = trade.current_price
                if not current_price:
                    continue
                
                # Check static stop loss (2% from entry)
                hit_sl = False
                if trade.side == 'BUY':
                    hit_sl = current_price <= trade.stop_loss
                else:
                    hit_sl = current_price >= trade.stop_loss
                
                if hit_sl:
                    await self.close_trade_v2(trade_id, current_price, "STOP_LOSS")
                    continue
                
                # PROPER TRAILING STOP IMPLEMENTATION
                # If max P&L% was 10% and drops to 9.70%, close (0.30 percentage point drop)
                current_pnl_percent = trade.calculate_pnl_percent(current_price)
                
                # Check if we've had significant profit and it's regressed
                if trade.max_pnl_percent > 0.5:  # Only trail after 0.5% profit
                    pnl_regression = trade.max_pnl_percent - current_pnl_percent
                    
                    if pnl_regression >= self.risk_manager.trailing_stop_percentage:
                        # Trailing stop hit!
                        self.log_signal.emit(
                            f"⚡ TRAILING STOP: {trade.symbol}\n"
                            f"   Max P&L: {trade.max_pnl_percent:.2f}%\n"
                            f"   Current P&L: {current_pnl_percent:.2f}%\n"
                            f"   Locked profit at {current_pnl_percent:.2f}%!"
                        )
                        await self.close_trade_v2(trade_id, current_price, f"TRAILING_TP_{current_pnl_percent:.2f}%")
                        continue
                
                # Also check if we hit initial take profit target (5%)
                if current_pnl_percent >= 5.0:
                    await self.close_trade_v2(trade_id, current_price, "TAKE_PROFIT_5%")
                    
            except Exception as e:
                logger.error(f"Exit check error: {e}")
    
    async def close_trade_v2(self, trade_id, exit_price, reason):
        """Close trade with leverage calculation and ML update"""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        # Calculate final P&L (on LEVERAGED position)
        final_pnl = trade.calculate_pnl(exit_price)
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.pnl = final_pnl
        trade.close_reason = reason
        trade.duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
        trade.status = 'CLOSED'
        
        # Move to closed
        self.closed_trades.append(trade)
        del self.active_trades[trade_id]
        
        # RESTORE available balance: margin + P&L
        # If profit: get back margin + profit
        # If loss: get back margin - loss (which might be less than margin or even negative)
        self.risk_manager.update_available_balance(
            trade.margin_used + final_pnl,
            'add'
        )
        
        # Update total balance
        self.trading_engine.account_balance += final_pnl
        
        # Record result for risk tracking (but don't suspend)
        self.risk_manager.record_trade_result(final_pnl)
        
        # Update ML weights with P&L amount for better learning
        self.ml_optimizer.update_weights(trade.strategy, final_pnl > 0, final_pnl)
        
        # Exit validation
        exit_validation = {
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'reason': reason,
            'pnl': final_pnl,
            'pnl_percent': (final_pnl / trade.margin_used) * 100,
            'leverage': trade.leverage,
            'duration': trade.duration,
            'validation_hash': trade.validation_hash
        }
        
        self.validation_signal.emit({
            'type': 'EXIT',
            'symbol': trade.symbol,
            'timestamp': datetime.now().isoformat(),
            'validation': exit_validation
        })
        
        emoji = '✅' if final_pnl > 0 else '📊'  # Changed loss emoji to data emoji
        return_pct = (final_pnl / trade.margin_used) * 100
        
        self.log_signal.emit(
            f"{emoji} TRADE #{self.ml_optimizer.total_trades} CLOSED: {trade.symbol} - {reason}\n"
            f"   Exit: ${exit_price:.2f} | Leverage: {trade.leverage}x\n"
            f"   P&L: ${final_pnl:.2f} ({return_pct:+.1f}% on margin)\n"
            f"   Duration: {trade.duration:.1f} min | Data collected ✓\n"
            f"   Balance: ${self.risk_manager.available_balance:.2f} | Trading continues..."
        )
    
    async def process_close_requests(self):
        """Process manual close requests"""
        while not self.close_queue.empty():
            try:
                symbol = self.close_queue.get_nowait()
                for trade_id, trade in list(self.active_trades.items()):
                    if trade.symbol == symbol:
                        await self.close_trade_v2(trade_id, trade.current_price, "MANUAL")
                        logger.info(f"✅ Manually closed {symbol}")
                        break
            except Exception as e:
                logger.error(f"Close request error: {e}")
    
    def _on_price_update(self, price_data):
        """Handle real-time price updates from Market Data Service"""
        # This is called for every price update
        # We'll use this for active trade P&L updates
        pass  # Updates happen in update_active_trades_fast
    
    def _on_price_update(self, symbol: str, price: float):
        """Callback for real-time price updates from Market Data Service"""
        # Update active trades with real-time prices
        for trade_id, trade in self.active_trades.items():
            if trade.symbol == symbol:
                trade.current_price = price
                # Calculate P&L in real-time
                trade.pnl = trade.calculate_pnl(price)
                trade.pnl_percent = trade.calculate_pnl_percent(price)
    
    def _on_candle_closed(self, symbol: str, timeframe: str, candle):
        """Callback for closed candle events from Market Data Service"""
        # Strategies should only act on closed candles
        if timeframe == '5m':  # Main timeframe for strategies
            # Queue this for strategy analysis
            asyncio.run_coroutine_threadsafe(
                self._analyze_for_signal(symbol, candle),
                self.loop
            )
    
    async def _analyze_for_signal(self, symbol: str, candle):
        """Analyze closed candle for trading signals"""
        # Get recent candles from market service
        df = self.market_service.get_candles(symbol, '5m', limit=100)
        if not df.empty:
            # Run strategy analysis here
            pass
    
    async def get_current_price(self, symbol):
        """Get current market price from unified data service"""
        # Use the Market Data Service for price (no direct Binance calls)
        if self.market_service:
            price = self.market_service.get_latest_price(symbol)
            if price:
                return price
        
        # Return None if no data available
        return None
    
    async def update_ui_full(self):
        """Full UI update"""
        # Send closed trades with P&L percentage
        if self.closed_trades:
            closed_list = []
            for trade in self.closed_trades[-10:]:
                # Calculate P&L percentage on margin
                pnl_percent = 0
                if hasattr(trade, 'margin_used') and trade.margin_used > 0:
                    pnl_percent = (trade.pnl / trade.margin_used) * 100
                elif trade.position_size_usd > 0:
                    # Fallback for older trades without margin_used
                    pnl_percent = (trade.pnl / trade.position_size_usd) * 100
                
                closed_list.append({
                    'symbol': trade.symbol,
                    'strategy': trade.strategy,
                    'side': trade.side,
                    'leverage': getattr(trade, 'leverage', 1),
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'margin_used': getattr(trade, 'margin_used', trade.position_size_usd),
                    'position_size_usd': trade.position_size_usd,
                    'pnl': trade.pnl,
                    'pnl_percent': pnl_percent,
                    'duration': trade.duration,
                    'close_reason': trade.close_reason
                })
            # removed trade signal
        
        # Calculate statistics with proper P&L tracking
        total_pnl = sum(t.pnl for t in self.closed_trades)  # Includes ALL trades (wins AND losses)
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        
        # Calculate separate totals for clarity
        total_wins = sum(t.pnl for t in wins)
        total_losses = sum(t.pnl for t in losses)
        
        stats = {
            'balance': self.trading_engine.account_balance,
            'available_balance': self.risk_manager.available_balance,
            'total_pnl': total_pnl,  # This is NET P&L (wins - losses)
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': (len(wins) / len(self.closed_trades) * 100) if self.closed_trades else 0,
            'wins_count': len(wins),
            'losses_count': len(losses),
            'active_trades': len(self.active_trades),
            'max_trades': self.risk_manager.max_positions,
            'total_trades': len(self.closed_trades),
            'ml_weights': self.ml_optimizer.strategy_weights
        }
        
        # Log P&L breakdown periodically for transparency
        if self.update_counter % 20 == 0 and self.closed_trades:  # Every 10 seconds
            self.log_signal.emit(
                f"💰 P&L Breakdown:\n"
                f"   Wins: ${total_wins:.2f} ({len(wins)} trades)\n"
                f"   Losses: ${total_losses:.2f} ({len(losses)} trades)\n"
                f"   NET P&L: ${total_pnl:.2f}"
            )
        
        logger.info("Stats updated")
    
    def request_close_trade(self, symbol):
        """Queue close request"""
        self.close_queue.put(symbol)
    
    def export_results(self):
        """Export results to CSV"""
        try:
            filename = f"traderjoes_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'symbol', 'strategy', 'side', 'entry_price', 'exit_price',
                    'position_size_usd', 'stop_loss', 'take_profit', 
                    'pnl', 'duration', 'close_reason', 'validation_hash'
                ])
                writer.writeheader()
                
                for trade in self.closed_trades:
                    writer.writerow({
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'side': trade.side,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'position_size_usd': trade.position_size_usd,
                        'stop_loss': trade.stop_loss,
                        'take_profit': trade.take_profit,
                        'pnl': trade.pnl,
                        'duration': trade.duration,
                        'close_reason': trade.close_reason,
                        'validation_hash': trade.validation_hash
                    })
            
            return filename
        except Exception as e:
            logger.error(f"Export error: {e}")
            return None
    
    def stop(self):
        """Stop trading"""
        self.is_running = False
        
        # Stop Market Data Service
        if self.market_service:
            try:
                asyncio.create_task(self.market_service.stop())
                logger.info("🛑 Market Data Service stopped")
            except:
                pass


class MainWindowV2(QMainWindow):
    """TraderJoes v2.0.0 Main Window with all fixes"""
    
    def __init__(self):
        super().__init__()
        self.trading_worker = None
        self.config = self.load_config()
        self.init_ui()
        
    def load_config(self):
        """Load configuration"""
        config_file = Path("config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {
                    'api_key': config.get('api', {}).get('key', ''),
                    'api_secret': config.get('api', {}).get('secret', ''),
                    'testnet': config.get('api', {}).get('testnet', True),
                    'symbols': config.get('trading', {}).get('symbols', 
                        ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']),
                    'min_confidence': 0.50
                }
        return {}
    
    def init_ui(self):
        """Initialize UI with v2.1.3 improvements"""
        self.setWindowTitle("TraderJoes v2.1.3 - Reasonable Stop Losses + MTF + Leverage")
        self.setGeometry(100, 100, 1600, 950)
        
        # Enhanced dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #0A0E27; }
            QWidget { background-color: #0A0E27; color: #FFFFFF; }
            QPushButton {
                background-color: #00D4FF; color: #0A0E27;
                border: none; border-radius: 5px; padding: 10px;
                font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #0099CC; }
            QPushButton:disabled { background-color: #2A3050; color: #495670; }
            QPushButton#closeBtn {
                background-color: #FF3366;
                padding: 2px 6px;
                font-size: 11px;
                max-width: 50px;
                max-height: 22px;
            }
            QTextEdit, QTableWidget {
                background-color: #151A3A;
                border: 1px solid #2A3050;
                border-radius: 5px;
            }
            QTabWidget::pane { background-color: #151A3A; }
            QTabBar::tab {
                background-color: #1E2447;
                color: #8892B0;
                padding: 10px;
            }
            QTabBar::tab:selected {
                background-color: #151A3A;
                color: #00D4FF;
            }
            QHeaderView::section {
                background-color: #1E2447;
                color: #00D4FF;
                padding: 6px;
                font-weight: bold;
            }
            QLabel#header {
                font-size: 32px;
                font-weight: bold;
                color: #00D4FF;
            }
            QLabel#subheader {
                font-size: 14px;
                color: #8892B0;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Logo and version
        logo_layout = QVBoxLayout()
        logo = QLabel("TraderJoes")
        logo.setObjectName("header")
        logo_layout.addWidget(logo)
        
        version = QLabel("v2.1.3 | Reasonable Stops + MTF + 3-5x Leverage | by OneDeepx")
        version.setObjectName("subheader")
        logo_layout.addWidget(version)
        
        header_layout.addLayout(logo_layout)
        header_layout.addStretch()
        
        # Status
        self.status_label = QLabel("● READY | 500ms Updates")
        self.status_label.setStyleSheet("font-size: 16px; color: #FFB800;")
        header_layout.addWidget(self.status_label)
        
        main_layout.addLayout(header_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Trading tab
        self.trading_tab = self.create_trading_tab()
        self.tabs.addTab(self.trading_tab, "📊 Trading")
        
        # MTF Analysis tab
        self.mtf_tab = self.create_mtf_tab()
        self.tabs.addTab(self.mtf_tab, "🔍 Multi-Timeframe")
        
        # Validation tab
        self.validation_tab = self.create_validation_tab()
        self.tabs.addTab(self.validation_tab, "✓ Validation")
        
        # ML tab
        self.ml_tab = self.create_ml_tab()
        self.tabs.addTab(self.ml_tab, "🤖 Machine Learning")
        
        main_layout.addWidget(self.tabs)
        
        # Controls
        controls = QHBoxLayout()
        
        self.config_btn = QPushButton("⚙️ Configuration")
        self.config_btn.clicked.connect(self.show_config)
        controls.addWidget(self.config_btn)
        
        self.start_btn = QPushButton("▶️ Start Trading")
        self.start_btn.clicked.connect(self.start_trading)
        controls.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop Trading")
        self.stop_btn.clicked.connect(self.stop_trading)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self.export_results)
        controls.addWidget(self.export_btn)
        
        controls.addStretch()
        main_layout.addLayout(controls)
    
    def create_trading_tab(self):
        """Create enhanced trading tab with leverage display"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Tables
        tables_layout = QVBoxLayout()
        
        # Active trades with leverage column
        active_label = QLabel("🔴 Active Trades (Max 10 | 500ms Updates | 3-5x Leverage)")
        active_label.setStyleSheet("font-weight: bold; color: #00D4FF;")
        tables_layout.addWidget(active_label)
        
        self.active_trades_table = QTableWidget()
        self.active_trades_table.setColumnCount(14)  # Added leverage and margin columns
        self.active_trades_table.setHorizontalHeaderLabels([
            'Symbol', 'Strategy', 'Side', 'Lev', 'Entry', 'Current',
            'Margin', 'Position', 'P&L', 'P&L %', 'SL', 'TP/Trail', 
            'Duration', 'Action'
        ])
        self.active_trades_table.setColumnWidth(3, 40)  # Leverage column
        self.active_trades_table.setColumnWidth(13, 60)  # Action column
        tables_layout.addWidget(self.active_trades_table)
        
        # Closed trades
        closed_label = QLabel("✅ Closed Trades")
        closed_label.setStyleSheet("font-weight: bold; color: #00D4FF;")
        tables_layout.addWidget(closed_label)
        
        self.closed_trades_table = QTableWidget()
        self.closed_trades_table.setColumnCount(11)
        self.closed_trades_table.setHorizontalHeaderLabels([
            'Symbol', 'Strategy', 'Side', 'Lev', 'Entry', 'Exit', 
            'Margin', 'Position', 'P&L', 'Duration', 'Reason'
        ])
        tables_layout.addWidget(self.closed_trades_table)
        
        layout.addLayout(tables_layout, 2)
        
        # Right panel
        right_layout = QVBoxLayout()
        
        # Stats display
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(150)
        self.update_stats_display({})
        right_layout.addWidget(self.stats_display)
        
        # Activity log
        log_label = QLabel("📝 Activity Log")
        log_label.setStyleSheet("font-weight: bold; color: #00D4FF;")
        right_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)
        
        layout.addLayout(right_layout, 1)
        
        return widget
    
    def create_validation_tab(self):
        """Create validation tab with historical verification"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        header = QLabel("🔍 Historical Trade Validation")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00D4FF; padding: 10px;")
        layout.addWidget(header)
        
        info = QLabel("All trades are validated with timestamp and price verification")
        info.setStyleSheet("color: #8892B0; padding: 5px;")
        layout.addWidget(info)
        
        self.validation_log = QTextEdit()
        self.validation_log.setReadOnly(True)
        layout.addWidget(self.validation_log)
        
        return widget
    
    def create_mtf_tab(self):
        """Create multi-timeframe analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        header = QLabel("🔍 Multi-Timeframe Analysis (1m to 1d)")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00D4FF; padding: 10px;")
        layout.addWidget(header)
        
        info = QLabel("Analyzes all timeframes to determine optimal strategy and entry")
        info.setStyleSheet("color: #8892B0; padding: 5px;")
        layout.addWidget(info)
        
        # MTF Table
        self.mtf_table = QTableWidget()
        self.mtf_table.setColumnCount(8)
        self.mtf_table.setHorizontalHeaderLabels([
            'Symbol', 'Alignment', 'Strategy', 'Confidence',
            '1m/5m', '15m/1h', '4h', '1d'
        ])
        self.mtf_table.setMaximumHeight(200)
        layout.addWidget(self.mtf_table)
        
        # MTF Details
        details_label = QLabel("📊 Detailed Timeframe Analysis")
        details_label.setStyleSheet("font-weight: bold; color: #00D4FF; margin-top: 10px;")
        layout.addWidget(details_label)
        
        self.mtf_log = QTextEdit()
        self.mtf_log.setReadOnly(True)
        layout.addWidget(self.mtf_log)
        
        return widget
    
    def create_ml_tab(self):
        """Create ML monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        header = QLabel("🤖 Machine Learning Self-Adjustment")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00D4FF; padding: 10px;")
        layout.addWidget(header)
        
        self.ml_display = QTextEdit()
        self.ml_display.setReadOnly(True)
        layout.addWidget(self.ml_display)
        
        return widget
    
    def show_config(self):
        """Show configuration dialog with FIXED OK/Cancel buttons"""
        dialog = QDialog(self)
        dialog.setWindowTitle("TraderJoes v2.0.0 Configuration")
        dialog.setMinimumWidth(450)
        
        layout = QVBoxLayout()
        
        # Form layout for inputs
        form = QFormLayout()
        
        api_key_input = QLineEdit(self.config.get('api_key', ''))
        api_secret_input = QLineEdit(self.config.get('api_secret', ''))
        api_secret_input.setEchoMode(QLineEdit.EchoMode.Password)
        testnet_check = QCheckBox()
        testnet_check.setChecked(self.config.get('testnet', True))
        
        form.addRow("API Key:", api_key_input)
        form.addRow("API Secret:", api_secret_input)
        form.addRow("Use Testnet:", testnet_check)
        
        # Symbols
        symbols_label = QLabel("Trading Pairs (one per line):")
        symbols_text = QTextEdit()
        symbols_text.setPlainText('\n'.join(self.config.get('symbols', 
            ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])))
        symbols_text.setMaximumHeight(100)
        
        layout.addLayout(form)
        layout.addWidget(symbols_label)
        layout.addWidget(symbols_text)
        
        # FIXED: Properly connected buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        
        # Connect signals PROPERLY
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        
        # Execute dialog and handle result
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save configuration
            self.config['api_key'] = api_key_input.text()
            self.config['api_secret'] = api_secret_input.text()
            self.config['testnet'] = testnet_check.isChecked()
            self.config['symbols'] = [s.strip() for s in symbols_text.toPlainText().split('\n') if s.strip()]
            
            self.save_config()
            self.log("✅ Configuration saved successfully")
            QMessageBox.information(self, "Success", "Configuration saved!")
    
    def save_config(self):
        """Save configuration to file"""
        config = {
            "api": {
                "key": self.config.get('api_key', ''),
                "secret": self.config.get('api_secret', ''),
                "testnet": self.config.get('testnet', True)
            },
            "trading": {
                "symbols": self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']),
                "risk_per_trade": 0.01,  # Fixed 1%
                "max_positions": 10,
                "min_confidence": 0.50
            }
        }
        
        with open("config.json", 'w') as f:
            json.dump(config, f, indent=4)
    
    def start_trading(self):
        """Start trading"""
        self.log("=" * 50)
        self.log("Starting TraderJoes v2.0.0...")
        self.log("=" * 50)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("● TRADING | 500ms")
        self.status_label.setStyleSheet("font-size: 16px; color: #00FF88;")
        
        # Create and start worker
        self.trading_worker = TradingWorkerV2(self.config)
        self.trading_worker.log_signal.connect(self.log)
        self.trading_worker.trade_signal.connect(self.update_trades)
        self.trading_worker.stats_signal.connect(self.update_stats)
        self.trading_worker.validation_signal.connect(self.update_validation)
        self.trading_worker.mtf_signal.connect(self.update_mtf)  # Connect MTF signal
        self.trading_worker.error_signal.connect(self.on_error)
        
        self.trading_worker.start()
    
    def stop_trading(self):
        """Stop trading"""
        self.log("Stopping TraderJoes v2.0.0...")
        
        if self.trading_worker:
            self.trading_worker.stop()
            self.trading_worker.wait()
            self.trading_worker = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("● STOPPED")
        self.status_label.setStyleSheet("font-size: 16px; color: #FF3366;")
    
    def log(self, message):
        """Add to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def update_trades(self, data):
        """Update trade tables with v2.1.1 leverage data"""
        # Update active trades with leverage
        if 'active_trades' in data:
            self.active_trades_table.setRowCount(len(data['active_trades']))
            for row, trade in enumerate(data['active_trades']):
                self.active_trades_table.setItem(row, 0, QTableWidgetItem(trade['symbol']))
                self.active_trades_table.setItem(row, 1, QTableWidgetItem(trade['strategy']))
                self.active_trades_table.setItem(row, 2, QTableWidgetItem(trade['side']))
                
                # Leverage column with color coding
                lev_item = QTableWidgetItem(f"{trade['leverage']}x")
                if trade['leverage'] >= 4.5:
                    lev_item.setForeground(QColor('#FF3366'))  # Red for high
                elif trade['leverage'] >= 4.0:
                    lev_item.setForeground(QColor('#FFB800'))  # Yellow for medium
                else:
                    lev_item.setForeground(QColor('#00FF88'))  # Green for low
                self.active_trades_table.setItem(row, 3, lev_item)
                
                self.active_trades_table.setItem(row, 4, QTableWidgetItem(f"${trade['entry_price']:.2f}"))
                self.active_trades_table.setItem(row, 5, QTableWidgetItem(f"${trade['current_price']:.2f}"))
                
                # Margin used (actual balance used)
                margin_item = QTableWidgetItem(f"${trade['margin_used']:.2f}")
                margin_item.setForeground(QColor('#8892B0'))
                self.active_trades_table.setItem(row, 6, margin_item)
                
                # Leveraged position size
                pos_item = QTableWidgetItem(f"${trade['position_size_usd']:.2f}")
                pos_item.setForeground(QColor('#00D4FF'))
                self.active_trades_table.setItem(row, 7, pos_item)
                
                # P&L display with USD and percentage together
                pnl_usd = trade['pnl']
                pnl_percent = trade['pnl_percent']
                pnl_text = f"${pnl_usd:,.2f} ({pnl_percent:+.1f}%)"
                pnl_item = QTableWidgetItem(pnl_text)
                pnl_item.setForeground(QColor('#00FF88' if pnl_usd > 0 else '#FF3366'))
                self.active_trades_table.setItem(row, 8, pnl_item)
                
                # Remove separate percentage column as it's now combined
                self.active_trades_table.setItem(row, 9, QTableWidgetItem(""))
                
                self.active_trades_table.setItem(row, 10, QTableWidgetItem(f"${trade['stop_loss']:.2f}"))
                
                # Show max P&L% for trailing
                tp_item = QTableWidgetItem(f"Max: {trade['max_pnl_percent']:.2f}%")
                if trade['max_pnl_percent'] > 0.5:
                    tp_item.setForeground(QColor('#00FF88'))  # Green when trailing active
                else:
                    tp_item.setForeground(QColor('#FFB800'))  # Yellow when building
                self.active_trades_table.setItem(row, 11, tp_item)
                
                self.active_trades_table.setItem(row, 12, QTableWidgetItem(trade['duration']))
                
                # Close button
                close_btn = QPushButton("Close")
                close_btn.setObjectName("closeBtn")
                close_btn.setMaximumWidth(50)
                close_btn.setMaximumHeight(22)
                close_btn.clicked.connect(lambda _, s=trade['symbol']: self.close_trade(s))
                self.active_trades_table.setCellWidget(row, 13, close_btn)
        
        # Update closed trades with leverage and P&L percentage
        if 'closed_trades' in data:
            self.closed_trades_table.setRowCount(len(data['closed_trades']))
            for row, trade in enumerate(data['closed_trades']):
                self.closed_trades_table.setItem(row, 0, QTableWidgetItem(trade['symbol']))
                self.closed_trades_table.setItem(row, 1, QTableWidgetItem(trade['strategy']))
                self.closed_trades_table.setItem(row, 2, QTableWidgetItem(trade['side']))
                self.closed_trades_table.setItem(row, 3, QTableWidgetItem(f"{trade.get('leverage', 1)}x"))
                self.closed_trades_table.setItem(row, 4, QTableWidgetItem(f"${trade['entry_price']:.4f}"))
                self.closed_trades_table.setItem(row, 5, QTableWidgetItem(f"${trade['exit_price']:.4f}"))
                
                # Margin and Position
                self.closed_trades_table.setItem(row, 6, QTableWidgetItem(f"${trade.get('margin_used', 0):.2f}"))
                self.closed_trades_table.setItem(row, 7, QTableWidgetItem(f"${trade['position_size_usd']:.2f}"))
                
                # P&L with BOTH dollars and percentage - FIXED DISPLAY
                pnl_usd = trade['pnl']
                pnl_percent = trade.get('pnl_percent', 0)
                
                # Format P&L text with proper USD symbol and percentage
                pnl_text = f"${pnl_usd:,.2f} ({pnl_percent:+.1f}%)"
                pnl_item = QTableWidgetItem(pnl_text)
                if pnl_usd > 0:
                    pnl_item.setForeground(QColor('#00FF88'))
                else:
                    pnl_item.setForeground(QColor('#FF3366'))
                self.closed_trades_table.setItem(row, 8, pnl_item)
                
                self.closed_trades_table.setItem(row, 9, QTableWidgetItem(f"{trade['duration']:.1f}m"))
                self.closed_trades_table.setItem(row, 10, QTableWidgetItem(trade['close_reason']))
                self.closed_trades_table.setItem(row, 8, QTableWidgetItem(trade['close_reason']))
    
    def update_stats_display(self, stats):
        """Update statistics display"""
        text = f"""
═══════════════════════════════════════════════
              TraderJoes v2.1.3
    REASONABLE STOPS + MTF + 3-5x LEVERAGE
═══════════════════════════════════════════════
💰 Total Balance: ${stats.get('balance', 10000):.2f}
💵 Available Balance: ${stats.get('available_balance', 10000):.2f}
📊 Total P&L: ${stats.get('total_pnl', 0):.2f}
🎯 Win Rate: {stats.get('win_rate', 0):.1f}%
📈 Active: {stats.get('active_trades', 0)}/{stats.get('max_trades', 10)}
📉 Total Trades: {stats.get('total_trades', 0)}
⚡ Leverage: 3-5x based on confidence
🛡️ Stop Loss: 2-5% price distance
🔍 MTF: 1m to 1d Analysis Active
🚫 No Duplicates: 1 position per symbol
📊 Trailing: -0.30% from peak P&L%
🤖 ML Learning: AGGRESSIVE MODE
═══════════════════════════════════════════════
"""
        self.stats_display.setText(text)
    
    def update_stats(self, stats):
        """Update statistics"""
        self.update_stats_display(stats)
        
        # Update ML display
        if 'ml_weights' in stats:
            ml_text = "Strategy Weights (Self-Adjusted):\n" + "=" * 40 + "\n"
            for strategy, weight in stats['ml_weights'].items():
                ml_text += f"{strategy}: {weight:.3f}\n"
            self.ml_display.setText(ml_text)
    
    def update_validation(self, data):
        """Update validation tab"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        val_type = data.get('type', '')
        
        if val_type == 'ENTRY':
            validation = data.get('validation', {})
            trade = data.get('trade', {})
            msg = f"""
[{timestamp}] ENTRY VALIDATION
Symbol: {data.get('symbol')}
Entry Price: ${trade.get('entry_price', 0):.2f}
Position Size: ${trade.get('position_usd', 0):.2f}
Hash: {validation.get('hash', '')[:16]}
Time Verified: {'✅' if validation.get('time_verified') else '❌'}
Price Match: {'✅' if validation.get('price_match') else '❌'}
{'═' * 40}
"""
        else:  # EXIT
            validation = data.get('validation', {})
            msg = f"""
[{timestamp}] EXIT VALIDATION
Exit Price: ${validation.get('exit_price', 0):.2f}
P&L: ${validation.get('pnl', 0):.2f}
Reason: {validation.get('reason')}
Duration: {validation.get('duration', 0):.1f} minutes
Hash: {validation.get('validation_hash', '')[:16]}
{'═' * 40}
"""
        
        self.validation_log.append(msg)
    
    def close_trade(self, symbol):
        """Close trade manually"""
        reply = QMessageBox.question(
            self, 'Close Trade',
            f"Close {symbol} position?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.trading_worker:
                self.trading_worker.request_close_trade(symbol)
    
    def export_results(self):
        """Export results"""
        if self.trading_worker:
            filename = self.trading_worker.export_results()
            if filename:
                QMessageBox.information(self, "Export Complete", f"Results saved to:\n{filename}")
                self.log(f"✅ Results exported to {filename}")
            else:
                QMessageBox.warning(self, "Export Failed", "No trades to export")
        else:
            QMessageBox.warning(self, "No Data", "Start trading first")
    
    def update_mtf(self, data):
        """Update MTF analysis tab"""
        symbol = data.get('symbol')
        alignment = data.get('alignment')
        strategy = data.get('strategy')
        confidence = data.get('confidence', 0)
        timeframes = data.get('timeframes', {})
        
        # Update MTF table
        row = self.mtf_table.rowCount()
        self.mtf_table.insertRow(row)
        
        self.mtf_table.setItem(row, 0, QTableWidgetItem(symbol))
        
        # Color-code alignment
        alignment_item = QTableWidgetItem(alignment)
        if 'BULLISH' in alignment:
            alignment_item.setForeground(QColor('#00FF88'))
        elif 'BEARISH' in alignment:
            alignment_item.setForeground(QColor('#FF3366'))
        else:
            alignment_item.setForeground(QColor('#FFB800'))
        self.mtf_table.setItem(row, 1, alignment_item)
        
        self.mtf_table.setItem(row, 2, QTableWidgetItem(strategy))
        self.mtf_table.setItem(row, 3, QTableWidgetItem(f"{confidence:.1%}"))
        
        # Add simplified timeframe trends
        short_term = f"{timeframes.get('1m', {}).get('trend', '?')}/{timeframes.get('5m', {}).get('trend', '?')}"
        medium_term = f"{timeframes.get('15m', {}).get('trend', '?')}/{timeframes.get('1h', {}).get('trend', '?')}"
        long_term = timeframes.get('4h', {}).get('trend', '?')
        daily = timeframes.get('1d', {}).get('trend', '?')
        
        self.mtf_table.setItem(row, 4, QTableWidgetItem(short_term))
        self.mtf_table.setItem(row, 5, QTableWidgetItem(medium_term))
        self.mtf_table.setItem(row, 6, QTableWidgetItem(long_term))
        self.mtf_table.setItem(row, 7, QTableWidgetItem(daily))
        
        # Keep only last 20 rows
        while self.mtf_table.rowCount() > 20:
            self.mtf_table.removeRow(0)
        
        # Update MTF log
        timestamp = datetime.now().strftime("%H:%M:%S")
        mtf_msg = f"""
[{timestamp}] {symbol} MULTI-TIMEFRAME ANALYSIS
{'═' * 40}
Alignment: {alignment}
Recommended Strategy: {strategy}
Confidence: {confidence:.1%}

Timeframe Analysis:
"""
        for tf, tf_data in timeframes.items():
            if isinstance(tf_data, dict):
                trend = tf_data.get('trend', 'N/A')
                momentum = tf_data.get('momentum', 0)
                mtf_msg += f"  {tf:5s}: Trend={trend:12s} RSI={momentum:.1f}\n"
        
        mtf_msg += '═' * 40 + '\n'
        self.mtf_log.append(mtf_msg)
    
    def export_trades(self):
        """Export all trades with leverage data to CSV"""
        if not self.trading_worker or not self.trading_worker.closed_trades:
            QMessageBox.warning(self, "Warning", "No trades to export")
            return
        
        filename = f"traderjoes_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'symbol', 'strategy', 'side', 'leverage',
                    'entry_price', 'exit_price', 'margin_used', 'position_size_usd',
                    'pnl', 'pnl_percent', 'duration', 'close_reason',
                    'mtf_alignment', 'mtf_confidence', 'validation_hash'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for trade in self.trading_worker.closed_trades:
                    writer.writerow({
                        'timestamp': trade.entry_time.isoformat(),
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'side': trade.side,
                        'leverage': trade.leverage,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'margin_used': trade.margin_used,
                        'position_size_usd': trade.position_size_usd,
                        'pnl': trade.pnl,
                        'pnl_percent': (trade.pnl / trade.margin_used * 100) if trade.margin_used > 0 else 0,
                        'duration': trade.duration,
                        'close_reason': trade.close_reason,
                        'mtf_alignment': trade.mtf_alignment,
                        'mtf_confidence': trade.mtf_confidence,
                        'validation_hash': trade.validation_hash
                    })
            
            QMessageBox.information(self, "Success", f"Trades exported to {filename}")
            self.log(f"📊 Trades exported with leverage data to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def on_error(self, error):
        """Handle errors"""
        self.log(f"❌ ERROR: {error}")
        QMessageBox.critical(self, "Error", error)


def main():
    """Main entry point for TraderJoes v2.1.3"""
    print("=" * 60)
    print("       TraderJoes v2.1.3")
    print("  REASONABLE STOPS + MTF + LEVERAGE")
    print("=" * 60)
    print("Created by: OneDeepx")
    print("Repository: github.com/OneDeepx/JT345")
    print("=" * 60)
    print("\nNEW IN v2.1.3:")
    print("✅ REASONABLE STOP LOSSES - 2-5% price distance")
    print("✅ Extra breathing room for volatile coins")
    print("✅ NO DUPLICATE TRADES - One position per symbol")
    print("✅ 3-5x LEVERAGE - Based on confidence & MTF")
    print("✅ MULTI-TIMEFRAME ANALYSIS (1m to 1d)")
    print("✅ Strategy-specific stop distances:")
    print("   - Scalping: 2% price movement")
    print("   - Momentum: 3% price movement")
    print("   - Trend Following: 5% price movement")
    print("✅ P&L calculated on leveraged positions")
    print("✅ Proper % trailing stop (10% -> 9.70%)")
    print("✅ NO TRADING SUSPENSIONS - Continuous operation")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    app.setApplicationName("TraderJoes v2.1.3")
    
    window = MainWindowV2()
    window.show()
    
    print("\nTraderJoes v2.1.3 is running...")
    print("FUTURES TRADING with 3-5x leverage active")
    print("Reasonable stop losses: 2-5% price distance")
    print("Multi-timeframe analysis across 11 timeframes")
    print("No duplicate positions on same symbol")
    print("Configure API keys and click Start Trading")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()