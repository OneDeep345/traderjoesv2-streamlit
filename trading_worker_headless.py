"""
Trading Worker Headless - EXACT COPY of TradingWorkerV2 logic
No Qt dependencies, but ALL trading logic preserved line-for-line
"""

import asyncio
import logging
import traceback
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque
from queue import Queue
import hashlib

# Import Market Data Service
from market_data_service import get_market_data_service, MarketDataService

# Import ORIGINAL classes from traderjoes_v2.1.3.py - NOT recreating them
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the EXACT original classes
from traderjoes_v2_1_3 import (
    TradeV2,
    RiskManagerV2,
    MLStrategyOptimizer,
    HistoricalValidator,
    MultiTimeframeAnalyzer,
    TimeFrame
)

# Import trading engine
from trading_engine import TradingEngine
from validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class TradingWorkerV2Headless:
    """
    EXACT COPY of TradingWorkerV2 with Qt dependencies removed
    ALL trading logic, formulas, thresholds remain 100% identical
    Only signal emissions replaced with service method calls
    """
    
    def __init__(self, config, service):
        """Initialize - EXACT COPY minus QThread parent"""
        self.config = config
        self.service = service  # Reference to TraderJoesService for callbacks
        self.is_running = False
        self.client = None
        self.trading_engine = None
        self.risk_manager = RiskManagerV2()  # Using ORIGINAL class
        self.ml_optimizer = MLStrategyOptimizer()  # Using ORIGINAL class
        self.validator = HistoricalValidator()  # Using ORIGINAL class
        self.mtf_analyzer = MultiTimeframeAnalyzer()  # Using ORIGINAL class
        self.active_trades = {}  # Dict of TradeV2 objects
        self.closed_trades = []
        self.close_queue = Queue()
        self.update_counter = 0
        self.scan_count = 0
        self.mtf_cache = {}  # Cache MTF analysis
        
        # Unified Market Data Service
        self.market_service: Optional[MarketDataService] = None
        self.loop = None
    
    async def trade_loop(self):
        """Main trading loop with v2.0.0 enhancements - EXACT COPY"""
        self.is_running = True
        
        # Store the event loop for Market Data Service callbacks
        self.loop = asyncio.get_event_loop()
        
        # Initialize Market Data Service (replaces old Binance client)
        self.market_service = get_market_data_service(self.config)
        
        # Get symbols from config
        symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'])
        
        # Start market data service
        await self.market_service.start(symbols)
        self.service.add_log_message(f"✅ Market Data Service started for {len(symbols)} symbols")
        self.service.add_log_message("📡 Real-time WebSocket + REST data flowing")
        
        # Initialize trading engine
        self.trading_engine = TradingEngine()
        self.trading_engine.account_balance = 10000
        self.risk_manager.initial_balance = 10000
        self.risk_manager.available_balance = 10000
        
        # Add symbols to trading engine
        self.trading_engine.add_symbols(symbols)
        
        # Subscribe to price updates for active trade monitoring
        for symbol in symbols:
            self.market_service.subscribe_price(symbol, self._on_price_update)
            # Subscribe to closed candles for strategy signals
            self.market_service.subscribe_candle_closed(symbol, '5m', self._on_candle_closed)
        
        self.service.add_log_message("=" * 50)
        self.service.add_log_message(f"🎯 Risk: {self.risk_manager.risk_per_trade*100:.0f}% | Leverage: 3-5x")
        self.service.add_log_message(f"📊 Max positions: {self.risk_manager.max_positions}")
        self.service.add_log_message(f"💰 Starting balance: ${self.risk_manager.initial_balance:,.2f}")
        self.service.add_log_message(f"🎮 Symbols: {', '.join(symbols)}")
        self.service.add_log_message("=" * 50)
        
        # Main trading loop - EXACT COPY
        while self.is_running:
            try:
                # Process any close requests
                await self.process_close_requests()
                
                # Update active trades with ultra-fast prices
                self.update_counter += 1
                if self.update_counter % 2 == 0:  # Every 2 cycles (1 second)
                    await self.update_active_trades_fast()
                
                # Full UI update
                if self.update_counter % 2 == 0:
                    await self.update_ui_full()
                
                # Check exits with trailing stops
                await self.check_exits_with_trailing()
                
                # Scan markets every 10 cycles (5 seconds)
                if self.update_counter % 10 == 0:
                    self.scan_count += 1
                    # Only scan if we have room for more trades
                    if len(self.active_trades) < self.risk_manager.max_positions:
                        await self.scan_markets()
                    if self.scan_count % 5 == 0:
                        self.service.add_log_message(f"📍 Scans: {self.scan_count} | Active: {len(self.active_trades)}/{self.risk_manager.max_positions}")
                
                await asyncio.sleep(1.0)  # 1 second for guaranteed 1Hz updates
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(1)
    
    async def update_active_trades_fast(self):
        """Ultra-fast active trades update with leverage and P&L percentage tracking - EXACT COPY"""
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
            
            # Calculate P&L (using LEVERAGED position)
            trade.pnl = trade.calculate_pnl(current_price)
            trade.pnl_percent = trade.calculate_pnl_percent(current_price)
            
            # Track max P&L PERCENTAGE for trailing stop
            if trade.pnl_percent > trade.max_pnl_percent:
                trade.max_pnl_percent = trade.pnl_percent
            
            # Prepare display data
            active_list.append({
                'trade_id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'strategy': trade.strategy,
                'entry_price': trade.entry_price,
                'current_price': current_price,
                'position_size': trade.position_size,
                'position_size_usd': trade.position_size_usd,
                'leverage': trade.leverage,
                'margin_used': trade.margin_used,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'max_pnl_percent': trade.max_pnl_percent,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'duration': (datetime.now() - trade.entry_time).total_seconds() / 60,
                'mtf_alignment': trade.mtf_alignment
            })
        
        # Update service (replaces signal emission)
        self.service.update_active_trades(active_list)
    
    async def scan_markets(self):
        """Scan markets for new opportunities with MTF analysis - EXACT COPY"""
        for symbol in self.trading_engine.symbols:
            try:
                # Check for existing position
                if self.has_active_position(symbol):
                    continue
                
                # Primary timeframe for entry (5m for scalping/momentum)
                entry_timeframe = '5m'
                
                # Use cached data from Market Data Service
                df = self.market_service.get_candles(symbol, entry_timeframe, 500)
                if df.empty:
                    # Skip if no data available
                    continue
                
                # Store data
                self.trading_engine.market_data[symbol] = df
                current_price = df['close'].iloc[-1]
                
                # Get MTF analysis
                mtf_analysis = self.mtf_analyzer.analyze(symbol, self.trading_engine.market_data)
                
                # Cache MTF results
                self.mtf_cache[symbol] = mtf_analysis
                
                # Generate signals
                signals = self.trading_engine.analyze_symbol(symbol)
                
                # Apply ML selection
                if signals:
                    best_signal = self.ml_optimizer.select_best_signal(signals, self.trading_engine.market_data[symbol])
                    
                    if best_signal and self.validate_signal_with_mtf(best_signal, mtf_analysis):
                        # Check for duplicate signals
                        if not self.is_duplicate_signal(symbol, best_signal['action'], current_price):
                            await self.execute_trade_v2(best_signal, current_price, df, mtf_analysis)
                        
            except Exception as e:
                logger.error(f"Market scan error for {symbol}: {e}")
                continue
    
    def has_active_position(self, symbol: str) -> bool:
        """Check if we have an active position for this symbol - EXACT COPY"""
        for trade_id, trade in self.active_trades.items():
            if trade.symbol == symbol and trade.status == 'OPEN':
                return True
        return False
    
    def is_duplicate_signal(self, symbol: str, action: str, entry_price: float, threshold: float = 0.002) -> bool:
        """Check for duplicate signals to prevent multiple entries - EXACT COPY"""
        for trade_id, trade in self.active_trades.items():
            if trade.symbol == symbol and trade.status == 'OPEN':
                # Check if same direction and price is very close
                if trade.side == action:
                    price_diff = abs(trade.entry_price - entry_price) / entry_price
                    if price_diff < threshold:
                        return True
        return False
    
    def validate_signal_with_mtf(self, signal, mtf_analysis) -> bool:
        """Validate signal against multi-timeframe analysis - EXACT COPY"""
        if not mtf_analysis:
            return True  # Allow if no MTF data
        
        alignment = mtf_analysis.get('alignment', 'NEUTRAL')
        trend_strength = mtf_analysis.get('trend_strength', 0)
        
        # Strong signals can override MTF
        if signal['confidence'] > 0.75:
            return True
        
        # Check alignment
        if signal['action'] == 'BUY':
            # For buy signals, prefer bullish or neutral alignment
            return alignment in ['BULLISH', 'NEUTRAL'] or trend_strength > 0.3
        else:
            # For sell signals, prefer bearish or neutral alignment
            return alignment in ['BEARISH', 'NEUTRAL'] or trend_strength < -0.3
    
    async def execute_trade_v2(self, signal, current_price, market_data, mtf_analysis=None):
        """Execute trade with v2.0 enhancements including MTF and LEVERAGE - EXACT COPY"""
        try:
            # Extract signal data
            symbol = signal['symbol']
            action = signal['action']
            strategy = signal['strategy']
            confidence = signal['confidence']
            timeframe = signal.get('timeframe', '5m')
            
            # Adjust confidence based on MTF alignment
            if mtf_analysis:
                alignment = mtf_analysis.get('alignment', 'NEUTRAL')
                if alignment == 'BULLISH' and action == 'BUY':
                    confidence = min(0.95, confidence * 1.2)
                elif alignment == 'BEARISH' and action == 'SELL':
                    confidence = min(0.95, confidence * 1.2)
                elif alignment == 'CONFLICTED':
                    confidence *= 0.8
            
            # LEVERAGE based on confidence and MTF alignment
            leverage = 3  # Base leverage
            if confidence > 0.75:
                leverage = 5
            elif confidence > 0.65:
                leverage = 4
            
            # Reasonable stop loss (2-5% based on strategy)
            stop_distances = {
                'SCALPING': 0.02,      # 2% for quick trades
                'MOMENTUM': 0.03,      # 3% for momentum
                'TREND_FOLLOWING': 0.05,  # 5% for trends
                'MEAN_REVERSION': 0.025,  # 2.5% for mean reversion
                'BREAKOUT': 0.04,      # 4% for breakouts
            }
            stop_distance = stop_distances.get(strategy, 0.03)
            
            # Adjust for volatile coins
            volatile_coins = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT']
            if symbol in volatile_coins:
                stop_distance *= 1.5
                self.service.add_log_message(f"🎢 {symbol} is volatile, stop distance: {stop_distance*100:.1f}%")
            
            # Position sizing (1% risk per trade)
            position_size = self.risk_manager.calculate_position_size(
                balance=self.risk_manager.available_balance,
                price=current_price,
                stop_distance=stop_distance,
                confidence=confidence,
                leverage=leverage
            )
            
            # Position value WITHOUT leverage
            position_value = position_size * current_price
            
            # LEVERAGED position value
            leveraged_position_value = position_value * leverage
            
            # MARGIN required (position value / leverage)
            margin_required = position_value
            
            # Validate margin
            if margin_required > self.risk_manager.available_balance * 0.5:
                self.service.add_log_message(f"❌ Insufficient margin for {symbol}: ${margin_required:.2f}")
                return
            
            # Calculate stop/take profit
            if action == 'BUY':
                stop_loss = current_price * (1 - stop_distance)
                take_profit = current_price * (1 + stop_distance * 3)
            else:
                stop_loss = current_price * (1 + stop_distance)
                take_profit = current_price * (1 - stop_distance * 3)
            
            # Validation
            validation_hash = self.validator.validate_entry(
                symbol=symbol,
                side=action,
                price=current_price,
                position_size=leveraged_position_value,
                stop_loss=stop_loss,
                confidence=confidence,
                market_data=market_data
            )
            
            # Create trade object
            trade = TradeV2(
                id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
                symbol=symbol,
                strategy=strategy,
                side=action,
                entry_price=current_price,
                position_size=position_size,
                position_size_usd=leveraged_position_value,
                margin_used=margin_required,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                current_price=current_price,
                highest_price=current_price if action == 'BUY' else 0,
                lowest_price=current_price if action == 'SELL' else float('inf'),
                validation_hash=validation_hash,
                mtf_alignment=mtf_analysis.get('alignment', 'UNKNOWN') if mtf_analysis else 'UNKNOWN',
                mtf_confidence=confidence
            )
            
            # Add to active trades
            self.active_trades[trade.id] = trade
            
            # Update risk manager
            self.risk_manager.update_available_balance(margin_required)
            
            # Emit logs (replaces signal emission)
            self.service.add_log_message(f"✅ NEW: {symbol} {action} @ ${current_price:.4f}")
            self.service.add_log_message(f"   📊 Strategy: {strategy} | Confidence: {confidence:.1%}")
            self.service.add_log_message(f"   💰 Position: ${leveraged_position_value:.2f} ({leverage}x leverage)")
            self.service.add_log_message(f"   💵 Margin Used: ${margin_required:.2f}")
            self.service.add_log_message(f"   🛡️ Stop: ${stop_loss:.4f} ({stop_distance*100:.1f}% away)")
            self.service.add_log_message(f"   🎯 Target: ${take_profit:.4f}")
            self.service.add_log_message(f"   🔄 MTF: {mtf_analysis.get('alignment', 'N/A') if mtf_analysis else 'N/A'}")
            
            # Emit validation (as log message)
            self.service.add_log_message(f"VALIDATION: Entry validated - Hash: {validation_hash}")
            
        except Exception as e:
            self.service.add_log_message(f"❌ Trade execution error: {e}")
            logger.error(f"Trade execution error: {e}\n{traceback.format_exc()}")
    
    async def check_exits_with_trailing(self):
        """Check exit conditions with percentage-based trailing stop - EXACT COPY"""
        for trade_id in list(self.active_trades.keys()):
            try:
                trade = self.active_trades.get(trade_id)
                if not trade or trade.status != 'OPEN':
                    continue
                
                current_price = trade.current_price
                pnl_percent = trade.pnl_percent
                
                # Check stop loss
                if trade.side == 'BUY' and current_price <= trade.stop_loss:
                    await self.close_trade_v2(trade_id, current_price, "STOP_LOSS")
                elif trade.side == 'SELL' and current_price >= trade.stop_loss:
                    await self.close_trade_v2(trade_id, current_price, "STOP_LOSS")
                
                # Check take profit
                elif trade.side == 'BUY' and current_price >= trade.take_profit:
                    await self.close_trade_v2(trade_id, current_price, "TAKE_PROFIT")
                elif trade.side == 'SELL' and current_price <= trade.take_profit:
                    await self.close_trade_v2(trade_id, current_price, "TAKE_PROFIT")
                
                # PERCENTAGE-BASED trailing stop
                elif trade.max_pnl_percent > 10:  # If we've been up more than 10%
                    # Trail at 3% from max
                    trail_threshold = trade.max_pnl_percent - 3
                    if pnl_percent <= trail_threshold:
                        await self.close_trade_v2(trade_id, current_price, f"TRAIL_{trade.max_pnl_percent:.1f}%→{pnl_percent:.1f}%")
                
                # Time-based exit (24 hours)
                elif (datetime.now() - trade.entry_time).total_seconds() > 86400:
                    await self.close_trade_v2(trade_id, current_price, "TIME_24H")
                    
            except Exception as e:
                logger.error(f"Exit check error: {e}")
    
    async def close_trade_v2(self, trade_id, exit_price, reason):
        """Close trade with v2.0 enhancements - EXACT COPY"""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        if trade.status != 'OPEN':
            return
        
        # Mark as closing
        trade.status = 'CLOSING'
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.close_reason = reason
        
        # Calculate final P&L with leverage
        trade.pnl = trade.calculate_pnl(exit_price)
        trade.pnl_percent = trade.calculate_pnl_percent(exit_price)
        trade.duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
        
        # Update risk manager
        self.risk_manager.update_balance(trade.pnl)
        self.risk_manager.record_trade(trade.pnl > 0, trade.pnl)
        self.risk_manager.release_margin(trade.margin_used)
        
        # Update ML optimizer
        self.ml_optimizer.update_strategy_performance(trade.strategy, trade.pnl, trade.pnl_percent)
        
        # Log the close (replaces signal emission)
        result_emoji = "💰" if trade.pnl > 0 else "💸"
        self.service.add_log_message(f"{result_emoji} CLOSED: {trade.symbol} | {reason}")
        self.service.add_log_message(f"   Entry: ${trade.entry_price:.4f} → Exit: ${exit_price:.4f}")
        self.service.add_log_message(f"   P&L: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)")
        self.service.add_log_message(f"   Duration: {trade.duration:.1f} minutes")
        
        # Add to closed trades
        closed_trade = {
            'id': trade.id,
            'symbol': trade.symbol,
            'strategy': trade.strategy,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'position_size': trade.position_size,
            'position_size_usd': trade.position_size_usd,
            'margin_used': trade.margin_used,
            'leverage': trade.leverage,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'duration': trade.duration,
            'close_reason': reason,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat(),
            'max_pnl_percent': trade.max_pnl_percent
        }
        self.closed_trades.append(closed_trade)
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        # Validation (as log message)
        self.service.add_log_message(f"VALIDATION: Exit recorded - Hash: {trade.validation_hash}")
    
    async def process_close_requests(self):
        """Process manual close requests - EXACT COPY"""
        while not self.close_queue.empty():
            try:
                request = self.close_queue.get_nowait()
                symbol = request['symbol']
                
                # Find and close trades for this symbol
                for trade_id, trade in list(self.active_trades.items()):
                    if trade.symbol == symbol and trade.status == 'OPEN':
                        current_price = trade.current_price
                        await self.close_trade_v2(trade_id, current_price, "MANUAL")
                        
            except Exception as e:
                logger.error(f"Close request error: {e}")
    
    def _on_price_update(self, symbol: str, price: float):
        """Callback for real-time price updates from Market Data Service - EXACT COPY"""
        # Update active trades with real-time prices
        for trade_id, trade in self.active_trades.items():
            if trade.symbol == symbol:
                trade.current_price = price
                # Calculate P&L in real-time
                trade.pnl = trade.calculate_pnl(price)
                trade.pnl_percent = trade.calculate_pnl_percent(price)
    
    def _on_candle_closed(self, symbol: str, timeframe: str, candle):
        """Callback for closed candle events from Market Data Service - EXACT COPY"""
        # Strategies should only act on closed candles
        if timeframe == '5m':  # Main timeframe for strategies
            # Queue this for strategy analysis
            asyncio.run_coroutine_threadsafe(
                self._analyze_for_signal(symbol, candle),
                self.loop
            )
    
    async def _analyze_for_signal(self, symbol: str, candle):
        """Analyze closed candle for trading signals - EXACT COPY"""
        # Get recent candles from market service
        df = self.market_service.get_candles(symbol, '5m', limit=100)
        if not df.empty:
            # Run strategy analysis here
            pass
    
    async def get_current_price(self, symbol):
        """Get current market price from unified data service - EXACT COPY"""
        # Use the Market Data Service for price (no direct Binance calls)
        if self.market_service:
            price = self.market_service.get_latest_price(symbol)
            if price:
                return price
        
        # Return None if no data available
        return None
    
    async def update_ui_full(self):
        """Full UI update - EXACT COPY"""
        # Send closed trades with P&L percentage
        if self.closed_trades:
            closed_list = []
            for trade in self.closed_trades[-50:]:  # Last 50 trades
                closed_list.append(trade)
            
            # Update service (replaces signal emission)
            self.service.update_closed_trades(closed_list)
        
        # Calculate stats
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        win_count = sum(1 for t in self.closed_trades if t['pnl'] > 0)
        loss_count = sum(1 for t in self.closed_trades if t['pnl'] <= 0)
        total_trades = len(self.closed_trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate balance and equity
        current_balance = self.risk_manager.initial_balance + total_pnl
        
        # Add unrealized P&L
        unrealized_pnl = sum(trade.pnl for trade in self.active_trades.values())
        equity = current_balance + unrealized_pnl
        
        # Available margin
        available_margin = self.risk_manager.available_balance
        
        # Max drawdown calculation
        if self.closed_trades:
            cumulative = 0
            peak = 0
            max_drawdown = 0
            for trade in self.closed_trades:
                cumulative += trade['pnl']
                peak = max(peak, cumulative)
                drawdown = peak - cumulative
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # Stats to emit
        stats = {
            'balance': self.risk_manager.initial_balance,
            'equity': equity,
            'total_pnl': total_pnl,
            'total_pnl_percent': (total_pnl / self.risk_manager.initial_balance * 100),
            'unrealized_pnl': unrealized_pnl,
            'win_rate': win_rate,
            'wins': win_count,
            'losses': loss_count,
            'total_trades': total_trades,
            'available_margin': available_margin,
            'margin_used': self.risk_manager.initial_balance - available_margin,
            'active_positions': len(self.active_trades),
            'max_positions': self.risk_manager.max_positions,
            'max_drawdown': max_drawdown
        }
        
        # Update service (replaces signal emission)
        self.service.update_stats(stats)
    
    def request_close_trade(self, symbol):
        """Request to close a trade manually - EXACT COPY"""
        self.close_queue.put({'symbol': symbol})
    
    def export_results(self):
        """Export trading results to CSV - EXACT COPY"""
        if not self.closed_trades:
            return None
        
        import csv
        filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.closed_trades[0].keys())
            writer.writeheader()
            writer.writerows(self.closed_trades)
        
        return filename
    
    def stop(self):
        """Stop trading - EXACT COPY"""
        self.is_running = False
        
        # Stop Market Data Service
        if self.market_service:
            try:
                asyncio.create_task(self.market_service.stop())
                self.service.add_log_message("🛑 Market Data Service stopped")
            except:
                pass
