"""
Advanced Crypto Futures Trading Bot Engine
Incorporates ML-driven strategy selection, multi-pair trading, and dynamic risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
from enum import Enum
import logging
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import talib
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingStrategy(Enum):
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    SCALPING = "scalping"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    HYBRID = "hybrid"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TRAILING_STOP = "TRAILING_STOP"


@dataclass
class TradingSignal:
    symbol: str
    strategy: TradingStrategy
    action: str  # 'BUY' or 'SELL'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    use_trailing: bool
    trailing_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    indicators: Dict = field(default_factory=dict)
    ml_score: float = 0.0


@dataclass
class Trade:
    id: str
    symbol: str
    strategy: TradingStrategy
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    use_trailing: bool
    trailing_percent: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    status: str = "OPEN"
    highest_price: float = 0.0
    lowest_price: float = 0.0


class MLStrategySelector:
    """Machine Learning model for strategy selection and parameter optimization"""
    
    def __init__(self):
        self.strategy_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.profit_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.performance_history = deque(maxlen=1000)
        
    def extract_features(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Extract technical features for ML model"""
        features = {}
        
        # Price features
        features['returns_1h'] = df['close'].pct_change(12).iloc[-1]
        features['returns_4h'] = df['close'].pct_change(48).iloc[-1]
        features['returns_24h'] = df['close'].pct_change(288).iloc[-1]
        
        # Volatility features
        features['volatility'] = df['close'].pct_change().std()
        features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        features['atr_ratio'] = features['atr'] / df['close'].iloc[-1]
        
        # Trend indicators
        features['rsi'] = talib.RSI(df['close'], timeperiod=14).iloc[-1]
        features['macd'], features['macd_signal'], _ = talib.MACD(df['close'])
        features['macd'] = features['macd'].iloc[-1] if isinstance(features['macd'], pd.Series) else features['macd']
        features['macd_signal'] = features['macd_signal'].iloc[-1] if isinstance(features['macd_signal'], pd.Series) else features['macd_signal']
        
        # Moving averages
        features['sma_20'] = talib.SMA(df['close'], timeperiod=20).iloc[-1]
        features['sma_50'] = talib.SMA(df['close'], timeperiod=50).iloc[-1]
        features['sma_200'] = talib.SMA(df['close'], timeperiod=200).iloc[-1]
        features['price_to_sma20'] = df['close'].iloc[-1] / features['sma_20']
        features['price_to_sma50'] = df['close'].iloc[-1] / features['sma_50']
        
        # Momentum indicators
        features['momentum'] = talib.MOM(df['close'], timeperiod=10).iloc[-1]
        features['roc'] = talib.ROC(df['close'], timeperiod=10).iloc[-1]
        features['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        # Volume features
        features['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        features['obv'] = talib.OBV(df['close'], df['volume']).iloc[-1]
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        features['bb_position'] = (df['close'].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        features['bb_width'] = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
        
        # Market microstructure
        features['spread'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        features['close_position'] = (df['close'].iloc[-1] - df['low'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1])
        
        self.feature_names = list(features.keys())
        return np.array(list(features.values())).reshape(1, -1)
    
    def train(self, historical_data: pd.DataFrame):
        """Train ML models on historical data"""
        try:
            # Prepare training data
            X, y_strategy, y_profit = self._prepare_training_data(historical_data)
            
            if len(X) < 100:
                logger.warning("Insufficient training data")
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_strat_train, y_strat_test, y_profit_train, y_profit_test = \
                train_test_split(X_scaled, y_strategy, y_profit, test_size=0.2, random_state=42)
            
            # Train strategy classifier
            self.strategy_classifier.fit(X_train, y_strat_train)
            strategy_accuracy = self.strategy_classifier.score(X_test, y_strat_test)
            logger.info(f"Strategy classifier accuracy: {strategy_accuracy:.2%}")
            
            # Train profit predictor
            self.profit_predictor.fit(X_train, y_profit_train)
            profit_r2 = self.profit_predictor.score(X_test, y_profit_test)
            logger.info(f"Profit predictor R²: {profit_r2:.2%}")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"ML training error: {e}")
    
    def _prepare_training_data(self, historical_data: pd.DataFrame) -> Tuple:
        """Prepare features and labels for training"""
        # This is a simplified version - in production, you'd have labeled historical trades
        X = []
        y_strategy = []
        y_profit = []
        
        # Generate synthetic training data based on historical patterns
        for i in range(100, len(historical_data) - 20):
            window = historical_data.iloc[i-100:i]
            features = self.extract_features(window, "TRAINING")
            
            # Simulate strategy selection based on market conditions
            future_return = (historical_data.iloc[i+20]['close'] - historical_data.iloc[i]['close']) / historical_data.iloc[i]['close']
            
            # Determine best strategy based on market conditions
            if abs(future_return) < 0.005:  # Small move
                strategy = TradingStrategy.SCALPING
            elif future_return > 0.02:  # Strong uptrend
                strategy = TradingStrategy.TREND_FOLLOWING
            elif abs(future_return) > 0.01:  # High volatility
                strategy = TradingStrategy.MOMENTUM
            else:
                strategy = TradingStrategy.MEAN_REVERSION
            
            X.append(features[0])
            y_strategy.append(strategy.value)
            y_profit.append(future_return)
        
        return np.array(X), np.array(y_strategy), np.array(y_profit)
    
    def select_strategy(self, features: np.ndarray) -> Tuple[TradingStrategy, float]:
        """Select optimal strategy based on current market conditions"""
        if not self.is_trained:
            # Default strategy selection based on simple rules
            return self._rule_based_selection(features)
        
        try:
            features_scaled = self.scaler.transform(features)
            
            # Get strategy prediction with confidence
            strategy_proba = self.strategy_classifier.predict_proba(features_scaled)[0]
            strategy_idx = np.argmax(strategy_proba)
            confidence = strategy_proba[strategy_idx]
            
            strategy_name = self.strategy_classifier.classes_[strategy_idx]
            strategy = TradingStrategy(strategy_name)
            
            # Predict expected profit
            expected_profit = self.profit_predictor.predict(features_scaled)[0]
            
            # Adjust confidence based on expected profit
            if expected_profit > 0:
                confidence = min(confidence * (1 + expected_profit), 1.0)
            
            return strategy, confidence
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self._rule_based_selection(features)
    
    def _rule_based_selection(self, features: np.ndarray) -> Tuple[TradingStrategy, float]:
        """Fallback rule-based strategy selection"""
        # Extract key features (assuming feature order)
        rsi = features[0, 6] if features.shape[1] > 6 else 50
        volatility = features[0, 3] if features.shape[1] > 3 else 0.01
        
        if rsi > 70 or rsi < 30:
            return TradingStrategy.MEAN_REVERSION, 0.7
        elif volatility > 0.02:
            return TradingStrategy.MOMENTUM, 0.6
        elif 0.005 < volatility < 0.01:
            return TradingStrategy.SCALPING, 0.65
        else:
            return TradingStrategy.TREND_FOLLOWING, 0.6


class TradingStrategyAnalyzer:
    """Implements various trading strategies with ML enhancement"""
    
    def __init__(self):
        self.ml_selector = MLStrategySelector()
        
    def analyze_trend_following(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Trend following strategy using multiple timeframe analysis"""
        try:
            # Calculate trend indicators
            sma_20 = talib.SMA(df['close'], timeperiod=20)
            sma_50 = talib.SMA(df['close'], timeperiod=50)
            sma_200 = talib.SMA(df['close'], timeperiod=200)
            
            # ADX for trend strength
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            
            current_price = df['close'].iloc[-1]
            
            # Trend conditions
            uptrend = (sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1] and
                      current_price > sma_20.iloc[-1] and
                      adx.iloc[-1] > 25)
            
            downtrend = (sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1] and
                        current_price < sma_20.iloc[-1] and
                        adx.iloc[-1] > 25)
            
            if uptrend and plus_di.iloc[-1] > minus_di.iloc[-1]:
                # Calculate dynamic SL/TP based on ATR
                atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
                stop_loss = current_price - (2.5 * atr)
                take_profit = current_price + (4 * atr)
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.TREND_FOLLOWING,
                    action="BUY",
                    confidence=min(adx.iloc[-1] / 100, 0.9),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=True,
                    trailing_percent=2.0,
                    indicators={
                        'adx': adx.iloc[-1],
                        'trend_strength': 'strong' if adx.iloc[-1] > 40 else 'moderate'
                    }
                )
            
            elif downtrend and minus_di.iloc[-1] > plus_di.iloc[-1]:
                atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
                stop_loss = current_price + (2.5 * atr)
                take_profit = current_price - (4 * atr)
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.TREND_FOLLOWING,
                    action="SELL",
                    confidence=min(adx.iloc[-1] / 100, 0.9),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=True,
                    trailing_percent=2.0,
                    indicators={
                        'adx': adx.iloc[-1],
                        'trend_strength': 'strong' if adx.iloc[-1] > 40 else 'moderate'
                    }
                )
                
        except Exception as e:
            logger.error(f"Trend following analysis error: {e}")
        
        return None
    
    def analyze_momentum(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Momentum trading strategy with multiple confirmation indicators"""
        try:
            # Momentum indicators
            rsi = talib.RSI(df['close'], timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
            mom = talib.MOM(df['close'], timeperiod=10)
            
            # Volume confirmation
            volume_sma = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1]
            
            current_price = df['close'].iloc[-1]
            
            # Bullish momentum conditions
            bullish_momentum = (
                rsi.iloc[-1] > 50 and rsi.iloc[-1] < 70 and
                rsi.iloc[-1] > rsi.iloc[-2] and  # RSI trending up
                macd.iloc[-1] > macd_signal.iloc[-1] and
                macd_hist.iloc[-1] > macd_hist.iloc[-2] and  # MACD histogram increasing
                stoch_k.iloc[-1] > stoch_d.iloc[-1] and
                mom.iloc[-1] > 0 and
                volume_ratio > 1.2  # Volume confirmation
            )
            
            # Bearish momentum conditions
            bearish_momentum = (
                rsi.iloc[-1] < 50 and rsi.iloc[-1] > 30 and
                rsi.iloc[-1] < rsi.iloc[-2] and  # RSI trending down
                macd.iloc[-1] < macd_signal.iloc[-1] and
                macd_hist.iloc[-1] < macd_hist.iloc[-2] and  # MACD histogram decreasing
                stoch_k.iloc[-1] < stoch_d.iloc[-1] and
                mom.iloc[-1] < 0 and
                volume_ratio > 1.2
            )
            
            if bullish_momentum:
                # Dynamic SL/TP for momentum trades
                atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
                
                # Calculate confidence based on indicator alignment
                confidence = 0.5
                confidence += 0.1 if abs(rsi.iloc[-1] - 60) < 10 else 0
                confidence += 0.1 if macd_hist.iloc[-1] > macd_hist.iloc[-3] else 0
                confidence += 0.1 if volume_ratio > 1.5 else 0
                confidence += 0.1 if mom.iloc[-1] > mom.iloc[-2] else 0
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.MOMENTUM,
                    action="BUY",
                    confidence=min(confidence, 0.85),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=True,
                    trailing_percent=1.5,
                    indicators={
                        'rsi': rsi.iloc[-1],
                        'macd_hist': macd_hist.iloc[-1],
                        'momentum': mom.iloc[-1],
                        'volume_ratio': volume_ratio
                    }
                )
            
            elif bearish_momentum:
                atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)
                
                confidence = 0.5
                confidence += 0.1 if abs(rsi.iloc[-1] - 40) < 10 else 0
                confidence += 0.1 if macd_hist.iloc[-1] < macd_hist.iloc[-3] else 0
                confidence += 0.1 if volume_ratio > 1.5 else 0
                confidence += 0.1 if mom.iloc[-1] < mom.iloc[-2] else 0
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.MOMENTUM,
                    action="SELL",
                    confidence=min(confidence, 0.85),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=True,
                    trailing_percent=1.5,
                    indicators={
                        'rsi': rsi.iloc[-1],
                        'macd_hist': macd_hist.iloc[-1],
                        'momentum': mom.iloc[-1],
                        'volume_ratio': volume_ratio
                    }
                )
                
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
        
        return None
    
    def analyze_scalping(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """High-frequency scalping strategy for quick profits"""
        try:
            # Use shorter timeframes for scalping
            ema_5 = talib.EMA(df['close'], timeperiod=5)
            ema_10 = talib.EMA(df['close'], timeperiod=10)
            
            # Bollinger Bands for volatility
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            
            # RSI for overbought/oversold
            rsi = talib.RSI(df['close'], timeperiod=7)  # Shorter period for scalping
            
            # Volume analysis
            volume_spike = df['volume'].iloc[-1] > df['volume'].rolling(10).mean().iloc[-1] * 1.5
            
            current_price = df['close'].iloc[-1]
            spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / current_price
            
            # Scalping conditions - looking for quick reversals
            long_scalp = (
                current_price < lower.iloc[-1] and  # Price at lower band
                ema_5.iloc[-1] > ema_5.iloc[-2] and  # EMA turning up
                rsi.iloc[-1] < 35 and  # Oversold
                volume_spike and  # Volume confirmation
                spread < 0.003  # Low spread for better entry
            )
            
            short_scalp = (
                current_price > upper.iloc[-1] and  # Price at upper band
                ema_5.iloc[-1] < ema_5.iloc[-2] and  # EMA turning down
                rsi.iloc[-1] > 65 and  # Overbought
                volume_spike and
                spread < 0.003
            )
            
            if long_scalp:
                # Tight SL/TP for scalping
                stop_loss = current_price * 0.997  # 0.3% stop loss
                take_profit = current_price * 1.005  # 0.5% take profit
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.SCALPING,
                    action="BUY",
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=False,  # No trailing for scalping
                    indicators={
                        'rsi': rsi.iloc[-1],
                        'bb_position': 'lower',
                        'spread': spread
                    }
                )
            
            elif short_scalp:
                stop_loss = current_price * 1.003
                take_profit = current_price * 0.995
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.SCALPING,
                    action="SELL",
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=False,
                    indicators={
                        'rsi': rsi.iloc[-1],
                        'bb_position': 'upper',
                        'spread': spread
                    }
                )
                
        except Exception as e:
            logger.error(f"Scalping analysis error: {e}")
        
        return None
    
    def analyze_mean_reversion(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Mean reversion strategy for range-bound markets"""
        try:
            # Calculate z-score for mean reversion
            sma_20 = talib.SMA(df['close'], timeperiod=20)
            std_20 = df['close'].rolling(window=20).std()
            z_score = (df['close'].iloc[-1] - sma_20.iloc[-1]) / std_20.iloc[-1]
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
            
            # RSI for confirmation
            rsi = talib.RSI(df['close'], timeperiod=14)
            
            # Check if market is range-bound (low ADX)
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            is_ranging = adx.iloc[-1] < 25
            
            current_price = df['close'].iloc[-1]
            
            # Mean reversion conditions
            long_reversion = (
                z_score < -2 and  # Price significantly below mean
                current_price < lower.iloc[-1] and
                rsi.iloc[-1] < 30 and
                is_ranging
            )
            
            short_reversion = (
                z_score > 2 and  # Price significantly above mean
                current_price > upper.iloc[-1] and
                rsi.iloc[-1] > 70 and
                is_ranging
            )
            
            if long_reversion:
                stop_loss = current_price * 0.99  # 1% stop loss
                take_profit = middle.iloc[-1]  # Target mean price
                
                confidence = 0.6
                confidence += 0.1 if abs(z_score) > 2.5 else 0
                confidence += 0.1 if rsi.iloc[-1] < 25 else 0
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.MEAN_REVERSION,
                    action="BUY",
                    confidence=min(confidence, 0.8),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=False,
                    indicators={
                        'z_score': z_score,
                        'rsi': rsi.iloc[-1],
                        'adx': adx.iloc[-1]
                    }
                )
            
            elif short_reversion:
                stop_loss = current_price * 1.01
                take_profit = middle.iloc[-1]
                
                confidence = 0.6
                confidence += 0.1 if abs(z_score) > 2.5 else 0
                confidence += 0.1 if rsi.iloc[-1] > 75 else 0
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.MEAN_REVERSION,
                    action="SELL",
                    confidence=min(confidence, 0.8),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=False,
                    indicators={
                        'z_score': z_score,
                        'rsi': rsi.iloc[-1],
                        'adx': adx.iloc[-1]
                    }
                )
                
        except Exception as e:
            logger.error(f"Mean reversion analysis error: {e}")
        
        return None
    
    def analyze_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Breakout strategy for explosive moves"""
        try:
            # Identify support/resistance levels
            high_20 = df['high'].rolling(window=20).max()
            low_20 = df['low'].rolling(window=20).min()
            
            # Volume analysis
            volume_avg = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_avg.iloc[-1]
            
            # ATR for volatility
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Keltner Channels
            ema_20 = talib.EMA(df['close'], timeperiod=20)
            kc_upper = ema_20 + (2 * atr)
            kc_lower = ema_20 - (2 * atr)
            
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            
            # Breakout conditions
            bullish_breakout = (
                current_price > high_20.iloc[-2] and  # Breaking previous high
                prev_close <= high_20.iloc[-2] and  # Wasn't above before
                current_price > kc_upper.iloc[-1] and  # Above Keltner Channel
                volume_ratio > 1.5  # Volume confirmation
            )
            
            bearish_breakout = (
                current_price < low_20.iloc[-2] and  # Breaking previous low
                prev_close >= low_20.iloc[-2] and
                current_price < kc_lower.iloc[-1] and
                volume_ratio > 1.5
            )
            
            if bullish_breakout:
                stop_loss = high_20.iloc[-2] - (0.5 * atr.iloc[-1])  # Below breakout level
                take_profit = current_price + (3 * atr.iloc[-1])  # 3x ATR target
                
                confidence = 0.65
                confidence += 0.1 if volume_ratio > 2 else 0
                confidence += 0.1 if current_price > kc_upper.iloc[-1] * 1.01 else 0
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.BREAKOUT,
                    action="BUY",
                    confidence=min(confidence, 0.85),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=True,
                    trailing_percent=2.5,
                    indicators={
                        'breakout_level': high_20.iloc[-2],
                        'volume_ratio': volume_ratio,
                        'atr': atr.iloc[-1]
                    }
                )
            
            elif bearish_breakout:
                stop_loss = low_20.iloc[-2] + (0.5 * atr.iloc[-1])
                take_profit = current_price - (3 * atr.iloc[-1])
                
                confidence = 0.65
                confidence += 0.1 if volume_ratio > 2 else 0
                confidence += 0.1 if current_price < kc_lower.iloc[-1] * 0.99 else 0
                
                return TradingSignal(
                    symbol=symbol,
                    strategy=TradingStrategy.BREAKOUT,
                    action="SELL",
                    confidence=min(confidence, 0.85),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_trailing=True,
                    trailing_percent=2.5,
                    indicators={
                        'breakout_level': low_20.iloc[-2],
                        'volume_ratio': volume_ratio,
                        'atr': atr.iloc[-1]
                    }
                )
                
        except Exception as e:
            logger.error(f"Breakout analysis error: {e}")
        
        return None
    
    def get_best_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Use ML to select best strategy and get trading signal"""
        try:
            # Extract features for ML model
            features = self.ml_selector.extract_features(df, symbol)
            
            # Get ML-recommended strategy
            recommended_strategy, ml_confidence = self.ml_selector.select_strategy(features)
            
            # Analyze all strategies
            signals = {
                TradingStrategy.TREND_FOLLOWING: self.analyze_trend_following(df, symbol),
                TradingStrategy.MOMENTUM: self.analyze_momentum(df, symbol),
                TradingStrategy.SCALPING: self.analyze_scalping(df, symbol),
                TradingStrategy.MEAN_REVERSION: self.analyze_mean_reversion(df, symbol),
                TradingStrategy.BREAKOUT: self.analyze_breakout(df, symbol)
            }
            
            # Filter out None signals
            valid_signals = {k: v for k, v in signals.items() if v is not None}
            
            if not valid_signals:
                return None
            
            # If ML recommends a strategy with a valid signal, prioritize it
            if recommended_strategy in valid_signals:
                signal = valid_signals[recommended_strategy]
                signal.ml_score = ml_confidence
                signal.confidence = (signal.confidence + ml_confidence) / 2
                return signal
            
            # Otherwise, return the signal with highest confidence
            best_signal = max(valid_signals.values(), key=lambda x: x.confidence)
            best_signal.ml_score = ml_confidence
            
            return best_signal
            
        except Exception as e:
            logger.error(f"Signal selection error: {e}")
            return None


class RiskManager:
    """Advanced risk management with ML-driven position sizing and stop-loss optimization"""
    
    def __init__(self, max_risk_per_trade: float = 0.01, max_positions: int = 10):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.active_trades: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0
        }
        
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                               stop_loss: float, confidence: float = 1.0) -> float:
        """Calculate position size using Kelly Criterion with confidence adjustment"""
        risk_amount = account_balance * self.max_risk_per_trade
        
        # Adjust risk based on confidence and performance
        if self.performance_metrics['win_rate'] > 0:
            kelly_fraction = self._calculate_kelly_fraction()
            risk_amount *= min(kelly_fraction * confidence, 1.0)
        
        price_diff = abs(entry_price - stop_loss)
        if price_diff > 0:
            position_size = risk_amount / price_diff
            return position_size
        
        return 0
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if self.performance_metrics['avg_loss'] == 0:
            return 0.5  # Default conservative fraction
        
        win_rate = self.performance_metrics['win_rate']
        win_loss_ratio = abs(self.performance_metrics['avg_win'] / self.performance_metrics['avg_loss'])
        
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Cap Kelly fraction for safety
        return max(min(kelly, 0.25), 0.01)
    
    def should_take_trade(self, signal: TradingSignal, account_balance: float) -> bool:
        """Determine if trade should be taken based on risk rules"""
        # Check max positions
        if len(self.active_trades) >= self.max_positions:
            return False
        
        # Check if already in position for this symbol
        if signal.symbol in self.active_trades:
            return False
        
        # Check correlation with existing positions
        correlation_risk = self._calculate_correlation_risk(signal.symbol)
        if correlation_risk > 0.7:  # Too correlated with existing positions
            return False
        
        # Check drawdown limits
        if self.performance_metrics['max_drawdown'] > 0.15:  # 15% max drawdown
            return False
        
        # Check confidence threshold
        if signal.confidence < 0.6:
            return False
        
        return True
    
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation with existing positions"""
        # Simplified correlation check
        # In production, you'd calculate actual price correlations
        similar_symbols = sum(1 for s in self.active_trades.keys() 
                            if s[:3] == symbol[:3])  # Same base currency
        
        return similar_symbols / max(len(self.active_trades), 1)
    
    def update_trailing_stops(self, current_prices: Dict[str, float]):
        """Update trailing stops for active trades"""
        for trade_id, trade in self.active_trades.items():
            if not trade.use_trailing:
                continue
            
            current_price = current_prices.get(trade.symbol)
            if not current_price:
                continue
            
            # Update highest/lowest prices
            if trade.action == "BUY":
                trade.highest_price = max(trade.highest_price, current_price)
                new_stop = trade.highest_price * (1 - trade.trailing_percent / 100)
                trade.stop_loss = max(trade.stop_loss, new_stop)
            else:  # SELL
                trade.lowest_price = min(trade.lowest_price, current_price)
                new_stop = trade.lowest_price * (1 + trade.trailing_percent / 100)
                trade.stop_loss = min(trade.stop_loss, new_stop)
    
    def check_exits(self, current_prices: Dict[str, float]) -> List[str]:
        """Check if any positions should be closed"""
        trades_to_close = []
        
        for trade_id, trade in self.active_trades.items():
            price = current_prices.get(trade.symbol)
            if not price:
                continue
            
            # Check stop loss
            if trade.action == "BUY":
                if price <= trade.stop_loss or price >= trade.take_profit:
                    trades_to_close.append(trade_id)
            else:  # SELL
                if price >= trade.stop_loss or price <= trade.take_profit:
                    trades_to_close.append(trade_id)
        
        return trades_to_close
    
    def close_trade(self, trade_id: str, exit_price: float) -> Trade:
        """Close a trade and update statistics"""
        trade = self.active_trades.pop(trade_id)
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = "CLOSED"
        
        # Calculate PnL
        if trade.action == "BUY":
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size
        
        self.trade_history.append(trade)
        self._update_performance_metrics()
        
        return trade
    
    def _update_performance_metrics(self):
        """Update performance metrics based on trade history"""
        if not self.trade_history:
            return
        
        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl < 0]
        
        self.performance_metrics['win_rate'] = len(wins) / len(self.trade_history)
        
        if wins:
            self.performance_metrics['avg_win'] = np.mean([t.pnl for t in wins])
        
        if losses:
            self.performance_metrics['avg_loss'] = np.mean([t.pnl for t in losses])
        
        # Calculate Sharpe ratio
        if len(self.trade_history) > 1:
            returns = [t.pnl for t in self.trade_history]
            self.performance_metrics['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in self.trade_history])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / np.maximum(running_max, 1)
        self.performance_metrics['max_drawdown'] = np.min(drawdown)
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        self.performance_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0


class TradingEngine:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.strategy_analyzer = TradingStrategyAnalyzer()
        self.risk_manager = RiskManager()
        self.symbols_to_trade = []
        self.market_data = {}
        self.is_running = False
        self.account_balance = 10000  # Always start with $10,000
        self.pending_opportunities = deque(maxlen=20)  # Queue of opportunities when at max positions
        
    def add_symbols(self, symbols: List[str]):
        """Add symbols to trade"""
        self.symbols_to_trade.extend(symbols)
        self.symbols_to_trade = list(set(self.symbols_to_trade))  # Remove duplicates
        
    async def update_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch latest market data for a symbol"""
        # This would connect to Binance API
        # For now, returning placeholder
        pass
    
    async def scan_markets(self):
        """Continuously scan markets for opportunities"""
        while self.is_running:
            try:
                for symbol in self.symbols_to_trade:
                    # Update market data
                    df = await self.update_market_data(symbol)
                    if df is not None:
                        self.market_data[symbol] = df
                        
                        # Get trading signal
                        signal = self.strategy_analyzer.get_best_signal(df, symbol)
                        
                        if signal and signal.confidence > 0.6:
                            # Check if we can take the trade
                            if self.risk_manager.should_take_trade(signal, self.account_balance):
                                await self.execute_trade(signal)
                            else:
                                # Add to pending opportunities
                                self.pending_opportunities.append(signal)
                
                # Check pending opportunities when positions close
                if len(self.risk_manager.active_trades) < self.risk_manager.max_positions:
                    await self.process_pending_opportunities()
                
                # Update trailing stops
                current_prices = {s: self.market_data[s]['close'].iloc[-1] 
                                for s in self.symbols_to_trade if s in self.market_data}
                self.risk_manager.update_trailing_stops(current_prices)
                
                # Check for exits
                trades_to_close = self.risk_manager.check_exits(current_prices)
                for trade_id in trades_to_close:
                    await self.close_position(trade_id, current_prices)
                
                await asyncio.sleep(5)  # Scan every 5 seconds
                
            except Exception as e:
                logger.error(f"Market scan error: {e}")
                await asyncio.sleep(10)
    
    async def execute_trade(self, signal: TradingSignal):
        """Execute a trade based on signal"""
        try:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                self.account_balance,
                signal.entry_price,
                signal.stop_loss,
                signal.confidence
            )
            
            if position_size <= 0:
                return
            
            # Create trade object
            trade = Trade(
                id=f"{signal.symbol}_{datetime.now().timestamp()}",
                symbol=signal.symbol,
                strategy=signal.strategy,
                entry_price=signal.entry_price,
                position_size=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                use_trailing=signal.use_trailing,
                trailing_percent=signal.trailing_percent,
                entry_time=datetime.now(),
                highest_price=signal.entry_price if signal.action == "BUY" else 0,
                lowest_price=signal.entry_price if signal.action == "SELL" else float('inf')
            )
            
            # Add to active trades
            self.risk_manager.active_trades[trade.id] = trade
            
            logger.info(f"Trade executed: {trade.id} - {signal.strategy.value} - {signal.action}")
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def close_position(self, trade_id: str, current_prices: Dict[str, float]):
        """Close a position"""
        try:
            trade = self.risk_manager.active_trades.get(trade_id)
            if not trade:
                return
            
            exit_price = current_prices.get(trade.symbol)
            if not exit_price:
                return
            
            closed_trade = self.risk_manager.close_trade(trade_id, exit_price)
            
            logger.info(f"Trade closed: {trade_id} - PnL: {closed_trade.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Position close error: {e}")
    
    async def process_pending_opportunities(self):
        """Process pending opportunities when slots become available"""
        while self.pending_opportunities and len(self.risk_manager.active_trades) < self.risk_manager.max_positions:
            signal = self.pending_opportunities.popleft()
            
            # Check if signal is still valid (not too old)
            if (datetime.now() - signal.timestamp).seconds < 60:  # 1 minute validity
                if self.risk_manager.should_take_trade(signal, self.account_balance):
                    await self.execute_trade(signal)
    
    def start(self):
        """Start the trading engine"""
        self.is_running = True
        logger.info("Trading engine started")
        
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        logger.info("Trading engine stopped")
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        return {
            'active_trades': len(self.risk_manager.active_trades),
            'total_trades': len(self.risk_manager.trade_history),
            'performance_metrics': self.risk_manager.performance_metrics,
            'pending_opportunities': len(self.pending_opportunities),
            'account_balance': self.account_balance
        }
