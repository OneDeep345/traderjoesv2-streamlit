"""
Unified Market Data Service for TraderJoes v2.1.3
As specified in ChatGPT instructions - single clean backend for all market data
Replaces ALL old networking, REST polling, and websocket implementations
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import websocket
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Price data structure"""
    symbol: str
    price: float
    timestamp: float
    volume: float = 0
    bid: float = 0
    ask: float = 0


@dataclass 
class CandleData:
    """Candle data structure"""
    symbol: str
    timeframe: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


class MarketDataService:
    """
    Unified Market Data Service - Single source of truth for all market data
    Replaces ALL old networking, REST polling, and websocket implementations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the market data service"""
        self.config = config
        api_config = config.get('api', {})
        self.testnet = api_config.get('testnet', False)
        
        # URLs
        if self.testnet:
            self.rest_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com"
        else:
            self.rest_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com"
        
        # Shared data cache
        self.price_cache: Dict[str, PriceData] = {}
        self.candle_cache: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=500)))
        self.cache_lock = threading.Lock()
        
        # Subscriptions
        self.price_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.candle_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # WebSocket management
        self.ws = None
        self.ws_thread = None
        self.ws_running = False
        
        # REST session
        self.session = None
        
        # Symbols to track
        self.symbols: List[str] = []
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        
        # Performance tracking
        self.last_update_time = {}
        self.update_count = 0
        self.reconnect_count = 0
        
        logger.info(f"MarketDataService initialized ({'Testnet' if self.testnet else 'Live'})")
    
    async def start(self, symbols: List[str]):
        """Start the market data service"""
        self.symbols = symbols
        logger.info(f"Starting MarketDataService for {len(symbols)} symbols")
        
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        
        # Seed initial data from REST
        await self._seed_initial_data()
        
        # Start WebSocket connection
        self._start_websocket()
        
        # Start slow refresh task
        asyncio.create_task(self._slow_refresh_task())
        
        logger.info("MarketDataService started successfully")
    
    async def stop(self):
        """Stop the market data service"""
        logger.info("Stopping MarketDataService...")
        
        # Stop WebSocket
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        
        # Close REST session
        if self.session:
            await self.session.close()
        
        logger.info("MarketDataService stopped")
    
    async def _seed_initial_data(self):
        """Seed initial data from REST API"""
        logger.info("Seeding initial data from REST...")
        
        tasks = []
        for symbol in self.symbols:
            # Get ticker price
            tasks.append(self._fetch_ticker(symbol))
            
            # Get initial candles for each timeframe
            for tf in self.timeframes:
                tasks.append(self._fetch_klines(symbol, tf))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Initial data seed complete: {success_count}/{len(tasks)} successful")
    
    async def _fetch_ticker(self, symbol: str) -> Optional[PriceData]:
        """Fetch ticker data from REST"""
        try:
            url = f"{self.rest_url}/fapi/v1/ticker/24hr"
            params = {'symbol': symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    price_data = PriceData(
                        symbol=symbol,
                        price=float(data['lastPrice']),
                        timestamp=time.time(),
                        volume=float(data['volume']),
                        bid=float(data['bidPrice']),
                        ask=float(data['askPrice'])
                    )
                    
                    with self.cache_lock:
                        self.price_cache[symbol] = price_data
                    
                    return price_data
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
        return None
    
    async def _fetch_klines(self, symbol: str, timeframe: str) -> bool:
        """Fetch kline/candle data from REST"""
        try:
            url = f"{self.rest_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': 100
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    candles = []
                    for kline in data:
                        candle = CandleData(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=kline[0] / 1000,
                            open=float(kline[1]),
                            high=float(kline[2]),
                            low=float(kline[3]),
                            close=float(kline[4]),
                            volume=float(kline[5]),
                            is_closed=True
                        )
                        candles.append(candle)
                    
                    with self.cache_lock:
                        self.candle_cache[symbol][timeframe].extend(candles)
                    
                    return True
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol} {timeframe}: {e}")
        return False
    
    def _start_websocket(self):
        """Start WebSocket connection"""
        self.ws_running = True
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
        logger.info("WebSocket thread started")
    
    def _run_websocket(self):
        """Run WebSocket connection in thread"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._process_ws_message(data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            if self.ws_running:
                self.reconnect_count += 1
                time.sleep(min(5 * self.reconnect_count, 30))
                if self.ws_running:
                    self._run_websocket()
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code}")
        
        def on_open(ws):
            logger.info(f"WebSocket connected for {len(self.symbols)} symbols")
            self.reconnect_count = 0
        
        # Build stream subscriptions
        streams = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            # Price streams
            streams.append(f"{symbol_lower}@aggTrade")
            streams.append(f"{symbol_lower}@bookTicker")
            # Kline streams for each timeframe
            for tf in self.timeframes:
                streams.append(f"{symbol_lower}@kline_{tf}")
        
        ws_url = f"{self.ws_url}/stream?streams={'/'.join(streams)}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        while self.ws_running:
            try:
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
                if self.ws_running:
                    time.sleep(5)
    
    def _process_ws_message(self, data: Dict):
        """Process WebSocket message"""
        stream = data.get('stream', '')
        stream_data = data.get('data', {})
        
        if not stream or not stream_data:
            return
        
        # Parse stream name
        parts = stream.split('@')
        if len(parts) < 2:
            return
        
        symbol = parts[0].upper()
        stream_type = parts[1]
        
        # Process based on stream type
        if stream_type == 'aggTrade' or stream_type == 'trade':
            self._process_trade(symbol, stream_data)
        elif stream_type == 'bookTicker':
            self._process_book_ticker(symbol, stream_data)
        elif 'kline' in stream_type:
            self._process_kline(symbol, stream_data)
    
    def _process_trade(self, symbol: str, data: Dict):
        """Process trade data"""
        price = float(data.get('p', 0))
        if price <= 0:
            return
        
        with self.cache_lock:
            if symbol not in self.price_cache:
                self.price_cache[symbol] = PriceData(
                    symbol=symbol,
                    price=price,
                    timestamp=time.time()
                )
            else:
                self.price_cache[symbol].price = price
                self.price_cache[symbol].timestamp = time.time()
        
        # Notify subscribers
        self._notify_price_subscribers(symbol, price)
        
        self.update_count += 1
        self.last_update_time[symbol] = time.time()
    
    def _process_book_ticker(self, symbol: str, data: Dict):
        """Process book ticker data"""
        bid = float(data.get('b', 0))
        ask = float(data.get('a', 0))
        
        if bid <= 0 or ask <= 0:
            return
        
        with self.cache_lock:
            if symbol in self.price_cache:
                self.price_cache[symbol].bid = bid
                self.price_cache[symbol].ask = ask
                self.price_cache[symbol].timestamp = time.time()
    
    def _process_kline(self, symbol: str, data: Dict):
        """Process kline data"""
        kline = data.get('k', {})
        if not kline:
            return
        
        timeframe = kline.get('i')
        is_closed = kline.get('x', False)
        
        candle = CandleData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=kline['t'] / 1000,
            open=float(kline['o']),
            high=float(kline['h']),
            low=float(kline['l']),
            close=float(kline['c']),
            volume=float(kline['v']),
            is_closed=is_closed
        )
        
        with self.cache_lock:
            candle_deque = self.candle_cache[symbol][timeframe]
            
            # Update or add candle
            if candle_deque and candle_deque[-1].timestamp == candle.timestamp:
                candle_deque[-1] = candle
            else:
                candle_deque.append(candle)
        
        # Notify subscribers if candle closed
        if is_closed:
            self._notify_candle_subscribers(symbol, timeframe, candle)
    
    async def _slow_refresh_task(self):
        """Periodic slow refresh of REST data"""
        while self.ws_running:
            await asyncio.sleep(60)  # Every minute
            
            try:
                # Refresh tickers for all symbols
                tasks = [self._fetch_ticker(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                logger.debug(f"Slow refresh complete. Updates: {self.update_count}")
            except Exception as e:
                logger.error(f"Slow refresh error: {e}")
    
    # PUBLIC API METHODS
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        with self.cache_lock:
            if symbol in self.price_cache:
                return self.price_cache[symbol].price
        return None
    
    def get_price_data(self, symbol: str) -> Optional[PriceData]:
        """Get full price data for a symbol"""
        with self.cache_lock:
            return self.price_cache.get(symbol)
    
    def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get candles as DataFrame"""
        with self.cache_lock:
            candles = list(self.candle_cache[symbol][timeframe])[-limit:]
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles])
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def subscribe_price(self, symbol: str, callback: Callable[[str, float], None]):
        """Subscribe to price updates"""
        self.price_subscribers[symbol].append(callback)
        logger.info(f"Price subscription added for {symbol}")
    
    def subscribe_candle_closed(self, symbol: str, timeframe: str, 
                               callback: Callable[[str, str, Any], None]):
        """Subscribe to closed candle events"""
        key = f"{symbol}:{timeframe}"
        self.candle_subscribers[key].append(callback)
        logger.info(f"Candle subscription added for {symbol} {timeframe}")
    
    def _notify_price_subscribers(self, symbol: str, price: float):
        """Notify price subscribers"""
        for callback in self.price_subscribers[symbol]:
            try:
                callback(symbol, price)
            except Exception as e:
                logger.error(f"Price callback error: {e}")
    
    def _notify_candle_subscribers(self, symbol: str, timeframe: str, candle: CandleData):
        """Notify candle subscribers"""
        key = f"{symbol}:{timeframe}"
        for callback in self.candle_subscribers[key]:
            try:
                callback(symbol, timeframe, candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all cached prices"""
        with self.cache_lock:
            return {symbol: data.price for symbol, data in self.price_cache.items()}
    
    def is_connected(self) -> bool:
        """Check if service is connected"""
        return self.ws_running and self.ws is not None


# Singleton instance
_market_data_service: Optional[MarketDataService] = None


def get_market_data_service(config: Dict[str, Any] = None) -> MarketDataService:
    """Get or create the market data service singleton"""
    global _market_data_service
    
    if _market_data_service is None:
        if config is None:
            raise ValueError("Config required for first initialization")
        _market_data_service = MarketDataService(config)
    
    return _market_data_service
