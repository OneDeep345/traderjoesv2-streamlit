"""
Validation Engine for Crypto Futures Trading Bot
Performs real-time validation of strategy execution, data integrity, and trading accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from collections import deque, defaultdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    STRATEGY_EXECUTION = "strategy_execution"
    PRICE_MATCHING = "price_matching"
    DATA_INTEGRITY = "data_integrity"
    RISK_COMPLIANCE = "risk_compliance"
    ORDER_EXECUTION = "order_execution"
    SLIPPAGE_ANALYSIS = "slippage_analysis"
    FEE_CALCULATION = "fee_calculation"
    ML_PREDICTION = "ml_prediction"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    id: str
    timestamp: datetime
    category: ValidationCategory
    status: ValidationStatus
    check_name: str
    expected_value: Any
    actual_value: Any
    deviation: float
    message: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'status': self.status.value,
            'check_name': self.check_name,
            'expected_value': str(self.expected_value),
            'actual_value': str(self.actual_value),
            'deviation': self.deviation,
            'message': self.message,
            'metadata': self.metadata
        }


@dataclass
class TradeValidation:
    """Complete validation record for a trade"""
    trade_id: str
    symbol: str
    strategy: str
    entry_time: datetime
    validation_checks: List[ValidationResult] = field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.PENDING
    confidence_score: float = 0.0
    
    def add_check(self, result: ValidationResult):
        self.validation_checks.append(result)
        self._update_overall_status()
    
    def _update_overall_status(self):
        """Update overall status based on individual checks"""
        if any(r.status == ValidationStatus.CRITICAL for r in self.validation_checks):
            self.overall_status = ValidationStatus.CRITICAL
        elif any(r.status == ValidationStatus.FAILED for r in self.validation_checks):
            self.overall_status = ValidationStatus.FAILED
        elif any(r.status == ValidationStatus.WARNING for r in self.validation_checks):
            self.overall_status = ValidationStatus.WARNING
        elif all(r.status == ValidationStatus.PASSED for r in self.validation_checks):
            self.overall_status = ValidationStatus.PASSED
        else:
            self.overall_status = ValidationStatus.PENDING


class RealisticExecutionSimulator:
    """Simulates realistic trade execution with slippage and fees"""
    
    def __init__(self):
        # Binance Futures fee structure
        self.maker_fee = 0.0002  # 0.02%
        self.taker_fee = 0.0004  # 0.04%
        
        # Slippage parameters (calibrated from historical data)
        self.base_slippage = 0.0001  # 0.01% base slippage
        self.volume_impact = 0.00005  # Additional slippage per volume unit
        self.volatility_multiplier = 1.5  # Slippage increases with volatility
        
        # Market impact factors
        self.large_order_threshold = 10000  # USDT
        self.large_order_impact = 0.0002  # Additional 0.02% for large orders
        
    def calculate_slippage(self, symbol: str, order_size: float, 
                          market_volatility: float, order_book_depth: Dict,
                          is_market_order: bool = True) -> float:
        """Calculate realistic slippage based on market conditions"""
        
        slippage = 0.0
        
        if is_market_order:
            # Base slippage
            slippage = self.base_slippage
            
            # Volatility impact
            slippage *= (1 + market_volatility * self.volatility_multiplier)
            
            # Order size impact
            if order_size > self.large_order_threshold:
                size_ratio = order_size / self.large_order_threshold
                slippage += self.large_order_impact * np.log1p(size_ratio)
            
            # Order book depth impact
            if order_book_depth:
                bid_ask_spread = self._calculate_spread(order_book_depth)
                slippage += bid_ask_spread / 2  # Half spread as slippage
                
                # Check if order would walk the book
                book_impact = self._calculate_book_walk_impact(
                    order_size, order_book_depth
                )
                slippage += book_impact
        
        return min(slippage, 0.01)  # Cap at 1% maximum slippage
    
    def _calculate_spread(self, order_book: Dict) -> float:
        """Calculate bid-ask spread from order book"""
        if 'bids' in order_book and 'asks' in order_book:
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
                return (best_ask - best_bid) / best_bid
        return 0.0001  # Default spread
    
    def _calculate_book_walk_impact(self, order_size: float, 
                                   order_book: Dict) -> float:
        """Calculate price impact from walking the order book"""
        impact = 0.0
        remaining_size = order_size
        
        # Simulate walking through order book levels
        levels = order_book.get('asks', []) if order_size > 0 else order_book.get('bids', [])
        
        if not levels:
            return 0.0
        
        base_price = levels[0][0]
        weighted_price = 0.0
        total_filled = 0.0
        
        for price, quantity in levels[:10]:  # Check first 10 levels
            fill_size = min(remaining_size, quantity)
            weighted_price += price * fill_size
            total_filled += fill_size
            remaining_size -= fill_size
            
            if remaining_size <= 0:
                break
        
        if total_filled > 0:
            avg_price = weighted_price / total_filled
            impact = abs(avg_price - base_price) / base_price
        
        return impact
    
    def calculate_fees(self, order_size: float, price: float, 
                      is_maker: bool = False) -> Dict[str, float]:
        """Calculate trading fees"""
        notional_value = order_size * price
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee_amount = notional_value * fee_rate
        
        return {
            'fee_rate': fee_rate,
            'fee_amount': fee_amount,
            'notional_value': notional_value,
            'fee_type': 'maker' if is_maker else 'taker'
        }
    
    def simulate_execution(self, intended_price: float, order_size: float,
                         side: str, symbol: str, market_conditions: Dict) -> Dict:
        """Simulate realistic order execution"""
        
        # Calculate slippage
        slippage = self.calculate_slippage(
            symbol=symbol,
            order_size=order_size,
            market_volatility=market_conditions.get('volatility', 0.01),
            order_book_depth=market_conditions.get('order_book', {}),
            is_market_order=market_conditions.get('is_market_order', True)
        )
        
        # Apply slippage to price
        if side == 'BUY':
            executed_price = intended_price * (1 + slippage)
        else:
            executed_price = intended_price * (1 - slippage)
        
        # Calculate fees
        fees = self.calculate_fees(
            order_size=order_size,
            price=executed_price,
            is_maker=not market_conditions.get('is_market_order', True)
        )
        
        # Total cost impact
        total_cost = order_size * executed_price + fees['fee_amount']
        
        return {
            'intended_price': intended_price,
            'executed_price': executed_price,
            'slippage': slippage,
            'slippage_cost': order_size * intended_price * slippage,
            'fees': fees,
            'total_cost': total_cost,
            'price_impact': (executed_price - intended_price) / intended_price,
            'execution_timestamp': datetime.now()
        }


class DataIntegrityValidator:
    """Validates data integrity and consistency"""
    
    def __init__(self):
        self.data_checksums = {}
        self.data_history = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_threshold = 3  # Standard deviations for anomaly detection
        
    def validate_price_data(self, symbol: str, df: pd.DataFrame) -> ValidationResult:
        """Validate price data integrity"""
        try:
            issues = []
            
            # Check for missing data
            if df.empty:
                return self._create_result(
                    ValidationStatus.FAILED,
                    "Empty dataframe",
                    None, None, 1.0
                )
            
            # Check for NaN values
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                issues.append(f"Found {nan_count} NaN values")
            
            # Check for negative prices
            if (df[['open', 'high', 'low', 'close']] < 0).any().any():
                issues.append("Negative prices detected")
            
            # Check OHLC consistency (High >= Low, etc.)
            inconsistent = (df['high'] < df['low']).sum()
            if inconsistent > 0:
                issues.append(f"{inconsistent} bars with high < low")
            
            inconsistent = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
            if inconsistent > 0:
                issues.append(f"{inconsistent} bars with invalid high")
            
            inconsistent = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
            if inconsistent > 0:
                issues.append(f"{inconsistent} bars with invalid low")
            
            # Check for price spikes (anomalies)
            returns = df['close'].pct_change()
            z_scores = np.abs(stats.zscore(returns.dropna()))
            anomalies = (z_scores > self.anomaly_threshold).sum()
            if anomalies > 0:
                issues.append(f"{anomalies} potential price anomalies")
            
            # Check timestamp consistency
            time_diffs = df.index.to_series().diff()
            irregular_times = (time_diffs != time_diffs.mode()[0]).sum()
            if irregular_times > 1:  # Allow for one irregularity
                issues.append(f"{irregular_times} irregular time intervals")
            
            # Generate checksum for data verification
            data_str = df.to_json()
            checksum = hashlib.sha256(data_str.encode()).hexdigest()[:8]
            
            if symbol in self.data_checksums:
                # Verify continuity with previous data
                last_checksum = self.data_checksums[symbol]
                # Additional continuity checks here
            
            self.data_checksums[symbol] = checksum
            
            # Determine status based on issues
            if not issues:
                status = ValidationStatus.PASSED
                message = "Data integrity check passed"
            elif len(issues) <= 2:
                status = ValidationStatus.WARNING
                message = f"Minor issues: {'; '.join(issues)}"
            else:
                status = ValidationStatus.FAILED
                message = f"Multiple issues: {'; '.join(issues)}"
            
            return ValidationResult(
                id=f"data_integrity_{symbol}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                category=ValidationCategory.DATA_INTEGRITY,
                status=status,
                check_name="price_data_validation",
                expected_value="Clean data",
                actual_value=issues if issues else "Clean",
                deviation=len(issues),
                message=message,
                metadata={'checksum': checksum, 'row_count': len(df)}
            )
            
        except Exception as e:
            logger.error(f"Data integrity validation error: {e}")
            return self._create_result(
                ValidationStatus.CRITICAL,
                f"Validation error: {str(e)}",
                None, None, 1.0
            )
    
    def validate_indicator_calculation(self, indicator_name: str,
                                      calculated_value: float,
                                      df: pd.DataFrame) -> ValidationResult:
        """Validate that indicators are calculated correctly"""
        try:
            import talib
            
            # Recalculate indicator independently
            recalculated = None
            
            if indicator_name == 'RSI':
                recalculated = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            elif indicator_name == 'MACD':
                macd, signal, _ = talib.MACD(df['close'])
                recalculated = macd.iloc[-1]
            elif indicator_name == 'SMA_20':
                recalculated = talib.SMA(df['close'], timeperiod=20).iloc[-1]
            elif indicator_name == 'ATR':
                recalculated = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            else:
                return self._create_result(
                    ValidationStatus.WARNING,
                    f"Unknown indicator: {indicator_name}",
                    None, calculated_value, 0.0
                )
            
            # Compare values
            if recalculated is not None and not np.isnan(recalculated):
                deviation = abs(calculated_value - recalculated) / abs(recalculated) if recalculated != 0 else 0
                
                if deviation < 0.001:  # Less than 0.1% deviation
                    status = ValidationStatus.PASSED
                    message = f"{indicator_name} calculation verified"
                elif deviation < 0.01:  # Less than 1% deviation
                    status = ValidationStatus.WARNING
                    message = f"{indicator_name} has minor deviation: {deviation:.4%}"
                else:
                    status = ValidationStatus.FAILED
                    message = f"{indicator_name} calculation mismatch: {deviation:.4%}"
                
                return ValidationResult(
                    id=f"indicator_{indicator_name}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    category=ValidationCategory.DATA_INTEGRITY,
                    status=status,
                    check_name=f"{indicator_name}_calculation",
                    expected_value=recalculated,
                    actual_value=calculated_value,
                    deviation=deviation,
                    message=message
                )
            
        except Exception as e:
            logger.error(f"Indicator validation error: {e}")
            
        return self._create_result(
            ValidationStatus.WARNING,
            f"Could not validate {indicator_name}",
            None, calculated_value, 0.0
        )
    
    def _create_result(self, status: ValidationStatus, message: str,
                      expected: Any, actual: Any, deviation: float) -> ValidationResult:
        """Helper to create validation result"""
        return ValidationResult(
            id=f"validation_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=ValidationCategory.DATA_INTEGRITY,
            status=status,
            check_name="data_validation",
            expected_value=expected,
            actual_value=actual,
            deviation=deviation,
            message=message
        )


class StrategyExecutionValidator:
    """Validates that strategies are executing according to their rules"""
    
    def __init__(self):
        self.strategy_rules = self._load_strategy_rules()
        self.execution_history = deque(maxlen=1000)
        
    def _load_strategy_rules(self) -> Dict:
        """Load strategy execution rules"""
        return {
            'TREND_FOLLOWING': {
                'required_indicators': ['ADX', 'SMA_20', 'SMA_50', 'SMA_200'],
                'entry_conditions': {
                    'ADX_min': 25,
                    'trend_alignment': True,  # Price > SMA20 > SMA50 > SMA200
                },
                'exit_conditions': {
                    'trailing_stop': True,
                    'trend_reversal': True
                }
            },
            'MOMENTUM': {
                'required_indicators': ['RSI', 'MACD', 'STOCH', 'MOM'],
                'entry_conditions': {
                    'RSI_range': (30, 70),
                    'MACD_signal': True,  # MACD > Signal
                    'volume_confirmation': 1.2  # Volume > 1.2x average
                },
                'exit_conditions': {
                    'momentum_exhaustion': True,
                    'trailing_stop': True
                }
            },
            'SCALPING': {
                'required_indicators': ['EMA_5', 'EMA_10', 'BB', 'RSI_7'],
                'entry_conditions': {
                    'price_at_bands': True,
                    'RSI_extreme': (30, 70),
                    'spread_max': 0.003
                },
                'exit_conditions': {
                    'fixed_targets': True,
                    'time_limit': 300  # 5 minutes
                }
            },
            'MEAN_REVERSION': {
                'required_indicators': ['Z_SCORE', 'BB', 'RSI', 'ADX'],
                'entry_conditions': {
                    'z_score_threshold': 2,
                    'ADX_max': 25,  # Low trend strength
                    'RSI_extreme': (30, 70)
                },
                'exit_conditions': {
                    'return_to_mean': True,
                    'stop_loss': True
                }
            },
            'BREAKOUT': {
                'required_indicators': ['ATR', 'VOLUME', 'RESISTANCE', 'SUPPORT'],
                'entry_conditions': {
                    'price_break': True,
                    'volume_surge': 1.5,
                    'ATR_expansion': True
                },
                'exit_conditions': {
                    'trailing_stop': True,
                    'momentum_loss': True
                }
            }
        }
    
    def validate_strategy_execution(self, signal: Dict, 
                                   market_data: pd.DataFrame) -> ValidationResult:
        """Validate that a trading signal follows strategy rules"""
        try:
            strategy = signal.get('strategy')
            if strategy not in self.strategy_rules:
                return ValidationResult(
                    id=f"strategy_exec_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    category=ValidationCategory.STRATEGY_EXECUTION,
                    status=ValidationStatus.WARNING,
                    check_name="strategy_validation",
                    expected_value="Known strategy",
                    actual_value=strategy,
                    deviation=1.0,
                    message=f"Unknown strategy: {strategy}"
                )
            
            rules = self.strategy_rules[strategy]
            violations = []
            
            # Check required indicators
            indicators = signal.get('indicators', {})
            for required_ind in rules['required_indicators']:
                if required_ind not in indicators:
                    violations.append(f"Missing indicator: {required_ind}")
            
            # Check entry conditions
            entry_conditions = rules['entry_conditions']
            
            # Strategy-specific validation
            if strategy == 'TREND_FOLLOWING':
                violations.extend(self._validate_trend_following(signal, market_data, entry_conditions))
            elif strategy == 'MOMENTUM':
                violations.extend(self._validate_momentum(signal, market_data, entry_conditions))
            elif strategy == 'SCALPING':
                violations.extend(self._validate_scalping(signal, market_data, entry_conditions))
            elif strategy == 'MEAN_REVERSION':
                violations.extend(self._validate_mean_reversion(signal, market_data, entry_conditions))
            elif strategy == 'BREAKOUT':
                violations.extend(self._validate_breakout(signal, market_data, entry_conditions))
            
            # Determine validation status
            if not violations:
                status = ValidationStatus.PASSED
                message = f"{strategy} execution validated successfully"
            elif len(violations) <= 1:
                status = ValidationStatus.WARNING
                message = f"Minor violation: {violations[0]}"
            else:
                status = ValidationStatus.FAILED
                message = f"Multiple violations: {'; '.join(violations[:3])}"
            
            return ValidationResult(
                id=f"strategy_exec_{strategy}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                category=ValidationCategory.STRATEGY_EXECUTION,
                status=status,
                check_name=f"{strategy}_execution",
                expected_value="All rules satisfied",
                actual_value=violations if violations else "All satisfied",
                deviation=len(violations),
                message=message,
                metadata={'strategy': strategy, 'violations': violations}
            )
            
        except Exception as e:
            logger.error(f"Strategy validation error: {e}")
            return ValidationResult(
                id=f"strategy_error_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                category=ValidationCategory.STRATEGY_EXECUTION,
                status=ValidationStatus.CRITICAL,
                check_name="strategy_validation",
                expected_value="Successful validation",
                actual_value=f"Error: {str(e)}",
                deviation=1.0,
                message=f"Validation error: {str(e)}"
            )
    
    def _validate_trend_following(self, signal: Dict, df: pd.DataFrame,
                                 conditions: Dict) -> List[str]:
        """Validate trend following strategy execution"""
        violations = []
        indicators = signal.get('indicators', {})
        
        # Check ADX threshold
        if 'adx' in indicators:
            if indicators['adx'] < conditions['ADX_min']:
                violations.append(f"ADX {indicators['adx']:.1f} below minimum {conditions['ADX_min']}")
        
        # Check trend alignment
        import talib
        sma_20 = talib.SMA(df['close'], 20).iloc[-1]
        sma_50 = talib.SMA(df['close'], 50).iloc[-1]
        sma_200 = talib.SMA(df['close'], 200).iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if signal.get('action') == 'BUY':
            if not (current_price > sma_20 > sma_50 > sma_200):
                violations.append("Trend alignment not satisfied for BUY")
        elif signal.get('action') == 'SELL':
            if not (current_price < sma_20 < sma_50 < sma_200):
                violations.append("Trend alignment not satisfied for SELL")
        
        return violations
    
    def _validate_momentum(self, signal: Dict, df: pd.DataFrame,
                          conditions: Dict) -> List[str]:
        """Validate momentum strategy execution"""
        violations = []
        indicators = signal.get('indicators', {})
        
        # Check RSI range
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            min_rsi, max_rsi = conditions['RSI_range']
            if not (min_rsi <= rsi <= max_rsi):
                violations.append(f"RSI {rsi:.1f} outside range {conditions['RSI_range']}")
        
        # Check volume confirmation
        if 'volume_ratio' in indicators:
            if indicators['volume_ratio'] < conditions['volume_confirmation']:
                violations.append(f"Insufficient volume confirmation: {indicators['volume_ratio']:.2f}")
        
        # Check MACD signal
        if 'macd_hist' in indicators:
            if signal.get('action') == 'BUY' and indicators['macd_hist'] <= 0:
                violations.append("MACD histogram not positive for BUY signal")
            elif signal.get('action') == 'SELL' and indicators['macd_hist'] >= 0:
                violations.append("MACD histogram not negative for SELL signal")
        
        return violations
    
    def _validate_scalping(self, signal: Dict, df: pd.DataFrame,
                         conditions: Dict) -> List[str]:
        """Validate scalping strategy execution"""
        violations = []
        indicators = signal.get('indicators', {})
        
        # Check spread
        if 'spread' in indicators:
            if indicators['spread'] > conditions['spread_max']:
                violations.append(f"Spread {indicators['spread']:.4f} exceeds maximum {conditions['spread_max']}")
        
        # Check RSI extremes
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if signal.get('action') == 'BUY' and rsi >= 35:
                violations.append(f"RSI {rsi:.1f} not oversold for BUY")
            elif signal.get('action') == 'SELL' and rsi <= 65:
                violations.append(f"RSI {rsi:.1f} not overbought for SELL")
        
        # Check Bollinger Bands position
        if 'bb_position' in indicators:
            if signal.get('action') == 'BUY' and indicators['bb_position'] != 'lower':
                violations.append("Price not at lower band for BUY")
            elif signal.get('action') == 'SELL' and indicators['bb_position'] != 'upper':
                violations.append("Price not at upper band for SELL")
        
        return violations
    
    def _validate_mean_reversion(self, signal: Dict, df: pd.DataFrame,
                                conditions: Dict) -> List[str]:
        """Validate mean reversion strategy execution"""
        violations = []
        indicators = signal.get('indicators', {})
        
        # Check Z-score
        if 'z_score' in indicators:
            z_score = abs(indicators['z_score'])
            if z_score < conditions['z_score_threshold']:
                violations.append(f"Z-score {z_score:.2f} below threshold {conditions['z_score_threshold']}")
        
        # Check ADX (should be low for ranging market)
        if 'adx' in indicators:
            if indicators['adx'] > conditions['ADX_max']:
                violations.append(f"ADX {indicators['adx']:.1f} too high for mean reversion")
        
        # Check RSI extremes
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if signal.get('action') == 'BUY' and rsi >= 30:
                violations.append(f"RSI {rsi:.1f} not oversold enough")
            elif signal.get('action') == 'SELL' and rsi <= 70:
                violations.append(f"RSI {rsi:.1f} not overbought enough")
        
        return violations
    
    def _validate_breakout(self, signal: Dict, df: pd.DataFrame,
                          conditions: Dict) -> List[str]:
        """Validate breakout strategy execution"""
        violations = []
        indicators = signal.get('indicators', {})
        
        # Check volume surge
        if 'volume_ratio' in indicators:
            if indicators['volume_ratio'] < conditions['volume_surge']:
                violations.append(f"Volume ratio {indicators['volume_ratio']:.2f} below required {conditions['volume_surge']}")
        
        # Check breakout level
        if 'breakout_level' in indicators:
            current_price = df['close'].iloc[-1]
            breakout_level = indicators['breakout_level']
            
            if signal.get('action') == 'BUY' and current_price <= breakout_level:
                violations.append(f"Price {current_price:.2f} not above breakout level {breakout_level:.2f}")
            elif signal.get('action') == 'SELL' and current_price >= breakout_level:
                violations.append(f"Price {current_price:.2f} not below breakout level {breakout_level:.2f}")
        
        # Check ATR expansion
        if 'atr' in indicators:
            # Should see volatility expansion on breakout
            recent_atr = df['close'].iloc[-20:].std() * np.sqrt(252)
            if indicators['atr'] <= recent_atr:
                violations.append("No ATR expansion detected for breakout")
        
        return violations


class ValidationEngine:
    """Main validation engine orchestrating all validation components"""
    
    def __init__(self):
        self.execution_simulator = RealisticExecutionSimulator()
        self.data_validator = DataIntegrityValidator()
        self.strategy_validator = StrategyExecutionValidator()
        
        self.validation_history = deque(maxlen=10000)
        self.trade_validations = {}
        self.statistics = defaultdict(lambda: {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'critical': 0
        })
        
        self.real_vs_intended = {
            'prices': [],
            'slippage': [],
            'fees': [],
            'total_impact': []
        }
        
    def validate_trade_entry(self, trade: Dict, signal: Dict, 
                            market_data: pd.DataFrame,
                            order_book: Dict = None) -> TradeValidation:
        """Comprehensive validation of trade entry"""
        
        trade_validation = TradeValidation(
            trade_id=trade.get('id', 'unknown'),
            symbol=trade.get('symbol', 'unknown'),
            strategy=signal.get('strategy', 'unknown'),
            entry_time=datetime.now()
        )
        
        # 1. Validate data integrity
        data_result = self.data_validator.validate_price_data(
            trade['symbol'], market_data
        )
        trade_validation.add_check(data_result)
        
        # 2. Validate strategy execution
        strategy_result = self.strategy_validator.validate_strategy_execution(
            signal, market_data
        )
        trade_validation.add_check(strategy_result)
        
        # 3. Validate price matching
        price_match_result = self._validate_price_matching(
            trade, signal, market_data
        )
        trade_validation.add_check(price_match_result)
        
        # 4. Simulate realistic execution
        execution_result = self._validate_execution_realism(
            trade, market_data, order_book
        )
        trade_validation.add_check(execution_result)
        
        # 5. Validate risk parameters
        risk_result = self._validate_risk_parameters(trade, signal)
        trade_validation.add_check(risk_result)
        
        # 6. Validate ML predictions if applicable
        if 'ml_score' in signal:
            ml_result = self._validate_ml_prediction(signal, market_data)
            trade_validation.add_check(ml_result)
        
        # Store validation
        self.trade_validations[trade['id']] = trade_validation
        self._update_statistics(trade_validation)
        
        return trade_validation
    
    def _validate_price_matching(self, trade: Dict, signal: Dict,
                                market_data: pd.DataFrame) -> ValidationResult:
        """Validate that entry price matches chart data"""
        
        chart_price = market_data['close'].iloc[-1]
        entry_price = trade.get('entry_price', 0)
        signal_price = signal.get('entry_price', 0)
        
        # Calculate deviations
        chart_deviation = abs(entry_price - chart_price) / chart_price if chart_price > 0 else 1.0
        signal_deviation = abs(entry_price - signal_price) / signal_price if signal_price > 0 else 1.0
        
        # Check if prices match within tolerance
        tolerance = 0.001  # 0.1% tolerance
        
        if chart_deviation < tolerance and signal_deviation < tolerance:
            status = ValidationStatus.PASSED
            message = "Price matching validated"
        elif chart_deviation < 0.005 and signal_deviation < 0.005:
            status = ValidationStatus.WARNING
            message = f"Minor price deviation: Chart {chart_deviation:.3%}, Signal {signal_deviation:.3%}"
        else:
            status = ValidationStatus.FAILED
            message = f"Price mismatch: Chart {chart_deviation:.3%}, Signal {signal_deviation:.3%}"
        
        return ValidationResult(
            id=f"price_match_{trade['id']}",
            timestamp=datetime.now(),
            category=ValidationCategory.PRICE_MATCHING,
            status=status,
            check_name="entry_price_matching",
            expected_value=f"Chart: {chart_price:.4f}, Signal: {signal_price:.4f}",
            actual_value=entry_price,
            deviation=max(chart_deviation, signal_deviation),
            message=message,
            metadata={
                'chart_price': chart_price,
                'signal_price': signal_price,
                'entry_price': entry_price
            }
        )
    
    def _validate_execution_realism(self, trade: Dict, 
                                   market_data: pd.DataFrame,
                                   order_book: Dict = None) -> ValidationResult:
        """Validate realistic execution with slippage and fees"""
        
        # Calculate market volatility
        returns = market_data['close'].pct_change()
        volatility = returns.std()
        
        # Simulate execution
        execution = self.execution_simulator.simulate_execution(
            intended_price=trade.get('entry_price', 0),
            order_size=trade.get('position_size', 0) * trade.get('entry_price', 0),
            side=trade.get('side', 'BUY'),
            symbol=trade.get('symbol', ''),
            market_conditions={
                'volatility': volatility,
                'order_book': order_book or {},
                'is_market_order': True
            }
        )
        
        # Store for analysis
        self.real_vs_intended['prices'].append({
            'intended': execution['intended_price'],
            'executed': execution['executed_price']
        })
        self.real_vs_intended['slippage'].append(execution['slippage'])
        self.real_vs_intended['fees'].append(execution['fees']['fee_amount'])
        
        # Validate execution costs
        total_impact = execution['slippage'] + execution['fees']['fee_rate']
        
        if total_impact < 0.001:  # Less than 0.1% total cost
            status = ValidationStatus.PASSED
            message = f"Execution costs within normal range: {total_impact:.3%}"
        elif total_impact < 0.005:  # Less than 0.5%
            status = ValidationStatus.WARNING
            message = f"Elevated execution costs: {total_impact:.3%}"
        else:
            status = ValidationStatus.FAILED
            message = f"Excessive execution costs: {total_impact:.3%}"
        
        return ValidationResult(
            id=f"execution_{trade['id']}",
            timestamp=datetime.now(),
            category=ValidationCategory.ORDER_EXECUTION,
            status=status,
            check_name="execution_realism",
            expected_value=trade.get('entry_price', 0),
            actual_value=execution['executed_price'],
            deviation=execution['price_impact'],
            message=message,
            metadata=execution
        )
    
    def _validate_risk_parameters(self, trade: Dict, signal: Dict) -> ValidationResult:
        """Validate risk management parameters"""
        
        violations = []
        
        # Check stop loss
        if 'stop_loss' not in trade or trade['stop_loss'] == 0:
            violations.append("No stop loss set")
        else:
            # Check stop loss distance
            entry_price = trade.get('entry_price', 0)
            stop_loss = trade['stop_loss']
            sl_distance = abs(stop_loss - entry_price) / entry_price if entry_price > 0 else 0
            
            if sl_distance > 0.05:  # More than 5%
                violations.append(f"Stop loss too far: {sl_distance:.2%}")
            elif sl_distance < 0.001:  # Less than 0.1%
                violations.append(f"Stop loss too tight: {sl_distance:.2%}")
        
        # Check take profit
        if 'take_profit' in trade and trade['take_profit'] > 0:
            tp_distance = abs(trade['take_profit'] - trade['entry_price']) / trade['entry_price']
            
            # Check risk-reward ratio
            if 'stop_loss' in trade and trade['stop_loss'] > 0:
                risk = abs(trade['entry_price'] - trade['stop_loss'])
                reward = abs(trade['take_profit'] - trade['entry_price'])
                rr_ratio = reward / risk if risk > 0 else 0
                
                if rr_ratio < 1.5:
                    violations.append(f"Poor risk-reward ratio: {rr_ratio:.2f}")
        
        # Check position sizing
        if 'position_size' in trade:
            # Validate against account risk rules
            # This would need account balance information
            pass
        
        if not violations:
            status = ValidationStatus.PASSED
            message = "Risk parameters validated"
        elif len(violations) <= 1:
            status = ValidationStatus.WARNING
            message = f"Risk warning: {violations[0]}"
        else:
            status = ValidationStatus.FAILED
            message = f"Risk violations: {'; '.join(violations)}"
        
        return ValidationResult(
            id=f"risk_{trade['id']}",
            timestamp=datetime.now(),
            category=ValidationCategory.RISK_COMPLIANCE,
            status=status,
            check_name="risk_parameters",
            expected_value="Compliant",
            actual_value=violations if violations else "Compliant",
            deviation=len(violations),
            message=message
        )
    
    def _validate_ml_prediction(self, signal: Dict, 
                               market_data: pd.DataFrame) -> ValidationResult:
        """Validate ML model predictions"""
        
        ml_score = signal.get('ml_score', 0)
        confidence = signal.get('confidence', 0)
        
        # Check if ML score aligns with confidence
        score_confidence_diff = abs(ml_score - confidence)
        
        if score_confidence_diff < 0.1:
            status = ValidationStatus.PASSED
            message = "ML prediction aligned with confidence"
        elif score_confidence_diff < 0.2:
            status = ValidationStatus.WARNING
            message = f"ML score and confidence divergence: {score_confidence_diff:.2f}"
        else:
            status = ValidationStatus.FAILED
            message = f"Significant ML/confidence mismatch: {score_confidence_diff:.2f}"
        
        return ValidationResult(
            id=f"ml_pred_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=ValidationCategory.ML_PREDICTION,
            status=status,
            check_name="ml_validation",
            expected_value=ml_score,
            actual_value=confidence,
            deviation=score_confidence_diff,
            message=message,
            metadata={'ml_score': ml_score, 'confidence': confidence}
        )
    
    def validate_trade_exit(self, trade: Dict, exit_price: float,
                           market_data: pd.DataFrame) -> List[ValidationResult]:
        """Validate trade exit conditions"""
        results = []
        
        # Validate exit price matches market
        chart_price = market_data['close'].iloc[-1]
        price_deviation = abs(exit_price - chart_price) / chart_price
        
        if price_deviation < 0.001:
            status = ValidationStatus.PASSED
            message = "Exit price matches market"
        elif price_deviation < 0.005:
            status = ValidationStatus.WARNING
            message = f"Minor exit price deviation: {price_deviation:.3%}"
        else:
            status = ValidationStatus.FAILED
            message = f"Exit price mismatch: {price_deviation:.3%}"
        
        results.append(ValidationResult(
            id=f"exit_price_{trade['id']}",
            timestamp=datetime.now(),
            category=ValidationCategory.PRICE_MATCHING,
            status=status,
            check_name="exit_price_validation",
            expected_value=chart_price,
            actual_value=exit_price,
            deviation=price_deviation,
            message=message
        ))
        
        # Validate PnL calculation
        entry_price = trade.get('entry_price', 0)
        position_size = trade.get('position_size', 0)
        expected_pnl = (exit_price - entry_price) * position_size
        
        if trade.get('side') == 'SELL':
            expected_pnl = -expected_pnl
        
        actual_pnl = trade.get('pnl', 0)
        pnl_deviation = abs(actual_pnl - expected_pnl) / abs(expected_pnl) if expected_pnl != 0 else 0
        
        if pnl_deviation < 0.001:
            status = ValidationStatus.PASSED
            message = "PnL calculation correct"
        else:
            status = ValidationStatus.WARNING
            message = f"PnL calculation deviation: {pnl_deviation:.3%}"
        
        results.append(ValidationResult(
            id=f"pnl_{trade['id']}",
            timestamp=datetime.now(),
            category=ValidationCategory.ORDER_EXECUTION,
            status=status,
            check_name="pnl_calculation",
            expected_value=expected_pnl,
            actual_value=actual_pnl,
            deviation=pnl_deviation,
            message=message
        ))
        
        return results
    
    def get_validation_summary(self) -> Dict:
        """Get summary of validation statistics"""
        
        summary = {
            'total_validations': sum(s['total'] for s in self.statistics.values()),
            'by_category': dict(self.statistics),
            'overall_health': self._calculate_health_score(),
            'recent_issues': self._get_recent_issues(),
            'execution_metrics': {
                'avg_slippage': np.mean(self.real_vs_intended['slippage']) if self.real_vs_intended['slippage'] else 0,
                'total_fees': sum(self.real_vs_intended['fees']),
                'price_accuracy': self._calculate_price_accuracy()
            }
        }
        
        return summary
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score"""
        total = sum(s['total'] for s in self.statistics.values())
        if total == 0:
            return 100.0
        
        passed = sum(s['passed'] for s in self.statistics.values())
        warnings = sum(s['warnings'] for s in self.statistics.values())
        
        # Weight: passed=100%, warning=70%, failed=0%, critical=-50%
        score = (passed * 100 + warnings * 70) / total
        
        return max(0, min(100, score))
    
    def _calculate_price_accuracy(self) -> float:
        """Calculate price execution accuracy"""
        if not self.real_vs_intended['prices']:
            return 100.0
        
        deviations = []
        for price_pair in self.real_vs_intended['prices']:
            if price_pair['intended'] > 0:
                deviation = abs(price_pair['executed'] - price_pair['intended']) / price_pair['intended']
                deviations.append(deviation)
        
        if deviations:
            avg_deviation = np.mean(deviations)
            # Convert to accuracy percentage (100% = perfect, 0% = >1% deviation)
            accuracy = max(0, (1 - avg_deviation * 100)) * 100
            return accuracy
        
        return 100.0
    
    def _get_recent_issues(self, limit: int = 10) -> List[Dict]:
        """Get recent validation issues"""
        issues = []
        
        for validation in reversed(self.validation_history):
            if validation.status in [ValidationStatus.FAILED, ValidationStatus.CRITICAL]:
                issues.append(validation.to_dict())
                if len(issues) >= limit:
                    break
        
        return issues
    
    def _update_statistics(self, trade_validation: TradeValidation):
        """Update validation statistics"""
        for check in trade_validation.validation_checks:
            self.statistics[check.category.value]['total'] += 1
            self.statistics[check.category.value][check.status.value] += 1
            self.validation_history.append(check)


# Import scipy for statistical functions
from scipy import stats
