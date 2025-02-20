

from app.viewmodels.api.market_data import get_account_balance, fetch_historical_data, execute_kraken_trade


import os

from app.models.shared_models import Strategy
from app.viewmodels.api.kraken_api import KrakenFuturesAPI


import logging
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.others import DailyReturnIndicator


import time
import threading
from queue import Queue
from flask import current_app, request

logger = logging.getLogger(__name__)

# Global variables for market monitoring
market_monitor_active = False
monitor_thread = None
market_data_queue = Queue()
initial_price = None
current_strategy = None
last_market_conditions = None
strategy_change_threshold = 0.02  # 2% change threshold for major changes
minor_change_threshold = 0.005  # 0.5% change for minor adjustments
rsi_change_threshold = 5  # RSI change threshold
volume_change_threshold = 0.1  # 10% volume change threshold
pattern_confidence_threshold = 0.6  # Pattern confidence threshold

def detect_candlestick_patterns(data):
    """Detect candlestick patterns for market analysis."""
    patterns = []
    logger.info("Starting candlestick pattern detection...")
    
    # Calculate body and shadows
    data['body'] = data['close'] - data['open']
    data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
    data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
    data['body_size'] = abs(data['body'])
    data['total_range'] = data['high'] - data['low']
    
    # Doji pattern (indecision)
    doji = abs(data['body']) <= 0.1 * (data['high'] - data['low'])
    if doji.iloc[-1]:
        patterns.append(('Doji', -0.5 if data['body'].iloc[-2] > 0 else 0.5))
    
    # Hammer pattern (bullish reversal)
    hammer = (data['lower_shadow'] > 2 * abs(data['body'])) & (data['upper_shadow'] <= abs(data['body']))
    if hammer.iloc[-1]:
        patterns.append(('Hammer', 1))
    
    # Shooting Star pattern (bearish reversal)
    shooting_star = (data['upper_shadow'] > 2 * abs(data['body'])) & (data['lower_shadow'] <= abs(data['body']))
    if shooting_star.iloc[-1]:
        patterns.append(('Shooting Star', -1))
    
    # Doji Star Bullish (bullish reversal)
    if len(data) >= 2:
        prev_doji = abs(data['body'].iloc[-2]) <= 0.1 * (data['high'].iloc[-2] - data['low'].iloc[-2])
        curr_bullish = data['body'].iloc[-1] > 0
        if prev_doji and curr_bullish:
            patterns.append(('Doji Star Bullish', 1))
    
    # Bullish Engulfing pattern
    if len(data) >= 2:
        prev_body = data['body'].iloc[-2]
        curr_body = data['body'].iloc[-1]
        if prev_body < 0 and curr_body > abs(prev_body):
            patterns.append(('Bullish Engulfing', 1))
    
    # Bearish Engulfing pattern
    if len(data) >= 2:
        prev_body = data['body'].iloc[-2]
        curr_body = data['body'].iloc[-1]
        if prev_body > 0 and curr_body < -abs(prev_body):
            patterns.append(('Bearish Engulfing', -1))
    
    # Three White Soldiers (bullish continuation)
    if len(data) >= 3:
        last_three = data['body'].tail(3)
        if all(x > 0 for x in last_three) and all(last_three.iloc[i] >= last_three.iloc[i-1] * 0.9 for i in range(1, 3)):
            patterns.append(('Three White Soldiers', 1.5))
    
    # Three Black Crows (bearish continuation)
    if len(data) >= 3:
        last_three = data['body'].tail(3)
        if all(x < 0 for x in last_three) and all(abs(last_three.iloc[i]) >= abs(last_three.iloc[i-1]) * 0.9 for i in range(1, 3)):
            patterns.append(('Three Black Crows', -1.5))
    
    # Morning Star (bullish reversal)
    if len(data) >= 3:
        first_body = data['body'].iloc[-3]
        second_body = data['body'].iloc[-2]
        third_body = data['body'].iloc[-1]
        if (first_body < 0 and 
            abs(second_body) <= 0.1 * (data['high'].iloc[-2] - data['low'].iloc[-2]) and
            third_body > 0 and third_body > abs(first_body) * 0.5):
            patterns.append(('Morning Star', 1.5))
    
    # Evening Star (bearish reversal)
    if len(data) >= 3:
        first_body = data['body'].iloc[-3]
        second_body = data['body'].iloc[-2]
        third_body = data['body'].iloc[-1]
        if (first_body > 0 and 
            abs(second_body) <= 0.1 * (data['high'].iloc[-2] - data['low'].iloc[-2]) and
            third_body < 0 and abs(third_body) > first_body * 0.5):
            patterns.append(('Evening Star', -1.5))
    
    # Piercing Line (bullish reversal)
    if len(data) >= 2:
        prev_body = data['body'].iloc[-2]
        curr_body = data['body'].iloc[-1]
        if (prev_body < 0 and curr_body > 0 and
            data['close'].iloc[-1] > data['open'].iloc[-2] + abs(prev_body) * 0.5):
            patterns.append(('Piercing Line', 1))
    
    return patterns

def calculate_stop_loss_take_profit(data, position_type='long', atr_multiplier=2, trading_mode='spot'):
    """Calculate dynamic stop-loss and take-profit levels based on ATR."""
    # For spot trading, we don't calculate stop-loss and take-profit
    if trading_mode == 'spot':
        return None, None
    
    # Only calculate for futures trading
    if trading_mode == 'futures':
        atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close']).average_true_range()
        current_price = data['close'].iloc[-1]
        atr_value = atr.iloc[-1]
        
        if position_type == 'long':
            stop_loss = current_price - (atr_value * atr_multiplier)
            take_profit = current_price + (atr_value * atr_multiplier * 1.5)
        else:  # short position
            stop_loss = current_price + (atr_value * atr_multiplier)
            take_profit = current_price - (atr_value * atr_multiplier * 1.5)
        
        return stop_loss, take_profit
    
    return None, None

def monitor_market():
    # global market_monitor_active, initial_price, current_strategy, last_market_conditions
    with current_app.app_context():
        """Continuously monitor market conditions and adapt strategy in real-time."""
        global market_monitor_active, initial_price, current_strategy, last_market_conditions
        last_action = None
        last_btc_balance = 0.0
        last_strategy = None
        last_check_time = time.time()
        check_interval = 15  
        while market_monitor_active:
            current_time = time.time()
            if current_time - last_check_time < check_interval:
                time.sleep(1)
                continue
            last_check_time = current_time
            try:
                trading_mode = current_app.config.get("TRADING_MODE", "spot")
                if trading_mode == 'futures':
                    futures_api_key = os.environ.get("KRAKEN_FUTURES_API_KEY")
                    futures_private_key = os.environ.get("KRAKEN_FUTURES_PRIVATE_KEY")
                    if not futures_api_key or not futures_private_key:
                        time.sleep(5)
                        continue
                    futures_client = KrakenFuturesAPI()
                    tickers = futures_client.get_tickers("PI_XBTUSD")
                    if not tickers or 'tickers' not in tickers:
                        time.sleep(5)
                        continue
                    current_price = float(tickers['tickers'][0]['last'])
                    historical_data = pd.DataFrame(futures_client.get_history("PI_XBTUSD"))
                    if historical_data.empty:
                        time.sleep(5)
                        continue
                    accounts = futures_client.get_accounts()
                    if not accounts or 'accounts' not in accounts:
                        time.sleep(5)
                        continue   
                    available_margin = float(accounts['accounts'][0].get('auxiliary', {}).get('usd', 0))
                else:
                    historical_data = fetch_historical_data("XBTUSD", 60, int(time.time() - 3600))
                    if historical_data is not None and not historical_data.empty:
                        current_price = historical_data['close'].iloc[-1]
                if historical_data is not None and not historical_data.empty:
                    if initial_price is None:
                        initial_price = current_price
                    price_change = ((current_price - initial_price) / initial_price) * 100 if initial_price else 0
                    current_conditions = {
                        'price': current_price,
                        'patterns': detect_candlestick_patterns(historical_data),
                        'rsi': RSIIndicator(close=historical_data["close"]).rsi().iloc[-1],
                        'macd': MACD(close=historical_data["close"]).macd().iloc[-1],
                        'stoch_k': StochasticOscillator(high=historical_data["high"], 
                                                        low=historical_data["low"], 
                                                        close=historical_data["close"]).stoch().iloc[-1]
                    }
                    if last_market_conditions:
                        price_change_since_last = abs((current_conditions['price'] - last_market_conditions['price']) 
                                                    / last_market_conditions['price'])
                        if (price_change_since_last > strategy_change_threshold or
                            current_conditions['patterns'] != last_market_conditions['patterns'] or
                            abs(current_conditions['rsi'] - last_market_conditions['rsi']) > 5 or
                            (current_conditions['macd'] * last_market_conditions['macd'] < 0)):
                            current_strategy = None
                            last_action = None
                    last_market_conditions = current_conditions
                    formatted_current_price = f"${current_price:,.2f}"
                    formatted_initial_price = f"${initial_price:,.2f}" if initial_price else "-"
                    formatted_price_change = f"{price_change:+.2f}%" if initial_price else "-"
                    patterns = detect_candlestick_patterns(historical_data)
                    performance, action, strategy_name, strategy_desc = evaluate_strategy_performance(
                        "Dynamic Strategy", historical_data, "XBTUSDT")  # Use same symbol for both modes
                balance_data = get_account_balance()
                current_btc_balance = float(balance_data.get('XXBT', '0.0'))
                if (strategy_name != last_strategy or action != last_action) and action != "HOLD":
                    last_strategy = strategy_name
                    last_action = action
                    trade_size_type = current_app.config.get("TRADE_SIZE_TYPE", "fixed")
                    trade_size_value = float(current_app.config.get("TRADE_SIZE_VALUE", 10))
                    usdt_balance = float(balance_data.get('USDT', '0.0'))
                    if trade_size_type == "fixed":
                        if action.lower() == 'buy':
                            trade_volume = trade_size_value
                        else:
                            trade_volume = current_btc_balance if current_btc_balance > 0 else 0
                    else:
                        if action.lower() == 'buy':
                            trade_volume = (usdt_balance * trade_size_value) / 100
                        else: 
                            trade_volume = (current_btc_balance * trade_size_value) / 100 if current_btc_balance > 0 else 0
                    logger.info(f"Calculated trade volume: {trade_volume} for {trade_size_type} size of {trade_size_value}")
                    try:
                        if action.lower() == 'sell' and current_btc_balance < trade_volume:
                            logger.info(f"Insufficient BTC balance. Attempting to buy {trade_volume} BTC worth of USDT...")
                            buy_volume = trade_volume * 1.001  # Add a small buffer for fees
                            buy_result = execute_kraken_trade(
                                symbol="XBTUSDT",
                                action="buy",
                                volume=buy_volume,
                                ordertype="market"
                            )
                            logger.info(f"BTC purchased: {buy_result}")
                            time.sleep(2) 
                            balance_data = get_account_balance()
                            current_btc_balance = float(balance_data.get('XXBT', '0.0'))
                        if action.lower() == 'sell':
                            trade_volume = min(trade_volume, current_btc_balance)
                        trade_result = execute_kraken_trade(
                            symbol="XBTUSDT",
                            action=action.lower(),
                            volume=trade_volume,
                            ordertype="market"
                        )
                        logger.info(f"Trade executed: {trade_result}")
                        market_data_queue.put({
                            'trade_executed': True,
                            'trade_volume': trade_volume,
                            'trade_result': trade_result
                        })
                    except Exception as e:
                        logger.error(f"Error executing trade: {e}")
                        market_data_queue.put({
                            'trade_error': str(e)
                        })
                    stop_loss, take_profit = calculate_stop_loss_take_profit(historical_data, 
                        'long' if action == 'BUY' else 'short')
                    rsi = RSIIndicator(close=historical_data["close"]).rsi()
                    macd = MACD(close=historical_data["close"])
                    stoch = StochasticOscillator(high=historical_data["high"], low=historical_data["low"], close=historical_data["close"])
                    formatted_stop_loss = f"${stop_loss:,.2f}" if stop_loss else "-"
                    formatted_take_profit = f"${take_profit:,.2f}" if take_profit else "-"
                    formatted_rsi = f"{rsi.iloc[-1]:.2f}" if not pd.isna(rsi.iloc[-1]) else "-"
                    formatted_macd = f"{macd.macd().iloc[-1]:.2f}" if not pd.isna(macd.macd().iloc[-1]) else "-"
                    formatted_stoch_k = f"{stoch.stoch().iloc[-1]:.2f}" if not pd.isna(stoch.stoch().iloc[-1]) else "-"
                    formatted_stoch_d = f"{stoch.stoch_signal().iloc[-1]:,.2f}" if not pd.isna(stoch.stoch_signal().iloc[-1]) else "-"
                    formatted_performance = f"{performance:+.2f}%" if performance else "-"
                    market_data_queue.put({
                        'current_price': formatted_current_price,
                        'initial_price': formatted_initial_price,
                        'price_change': formatted_price_change,
                        'action': action,
                        'strategy': strategy_name,
                        'description': strategy_desc,
                        'performance': formatted_performance,
                        'patterns': ', '.join([p[0] for p in patterns]) if patterns else 'No patterns detected',
                        'stop_loss': formatted_stop_loss,
                        'take_profit': formatted_take_profit,
                        'trading_mode': trading_mode,
                        'available_margin': f"${available_margin:,.2f}" if available_margin else "-",
                        'rsi': formatted_rsi,
                        'macd': formatted_macd,
                        'stochastic_k': formatted_stoch_k,
                        'stochastic_d': formatted_stoch_d
                    })
                balance_data = get_account_balance()
                if balance_data.get('error'):
                    pass
                else:
                    pass
                time.sleep(15)
            except Exception as e:
                logger.error(f"Error in market monitor: {e}")
                time.sleep(5)


def start_market_monitor():
    """Start the market monitoring thread."""
    global market_monitor_active, monitor_thread, initial_price
    
    if not market_monitor_active:
        market_monitor_active = True
        initial_price = None  # Reset initial price
        monitor_thread = threading.Thread(target=monitor_market)
        monitor_thread.daemon = True  # Thread will stop when main program exits
        monitor_thread.start()
        logger.info("Market monitor started")

def stop_market_monitor():
    """Stop the market monitoring thread."""
    global market_monitor_active, monitor_thread, initial_price
    
    if market_monitor_active:
        market_monitor_active = False
        if monitor_thread:
            monitor_thread.join(timeout=5)
        initial_price = None
        logger.info("Market monitor stopped")

def set_trading_mode(mode):
    """Set the trading mode (spot or futures)."""
    global trading_mode
    if mode in ['spot', 'futures']:
        trading_mode = mode
        if mode == 'futures':
            return "BTCUSD.M"  # Multi-Collateral Future symbol for Kraken
        return "XBTUSDT"  # Default spot symbol
    logger.info(f"Trading mode set to: {mode}")

def evaluate_strategy_performance(strategy_name, historical_data, symbol, use_custom_strategies=False):
    """
    Evaluate the performance of trading strategies using historical data.
    Returns performance score, recommended action, strategy name and description.
    """
    try:
        logger.info(f"Starting strategy evaluation for {symbol}")
        if historical_data is None or historical_data.empty:
            logger.error("No historical data available for strategy evaluation")
            return 0.0, "HOLD", "No Strategy", "No historical data available"
        
        # Get current price from historical data
        current_price = float(historical_data["close"].iloc[-1])
        logger.info(f"Current BTC price: ${current_price:,.2f}")
        
        # Initialize technical indicators
        data = historical_data.copy()
        
        # Moving Averages
        data["SMA_20"] = SMAIndicator(close=data["close"], window=20).sma_indicator()
        data["SMA_50"] = SMAIndicator(close=data["close"], window=50).sma_indicator()
        data["EMA_20"] = EMAIndicator(close=data["close"], window=20).ema_indicator()
        
        # RSI
        data["RSI"] = RSIIndicator(close=data["close"], window=14).rsi()
        
        # MACD
        macd = MACD(close=data["close"])
        data["MACD"] = macd.macd()
        data["MACD_signal"] = macd.macd_signal()
        data["MACD_diff"] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=data["close"])
        data["BB_upper"] = bollinger.bollinger_hband()
        data["BB_lower"] = bollinger.bollinger_lband()
        data["BB_middle"] = bollinger.bollinger_mavg()
        data["BB_width"] = (data["BB_upper"] - data["BB_lower"]) / data["BB_middle"]
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=data["high"], low=data["low"], close=data["close"])
        data["STOCH_k"] = stoch.stoch()
        data["STOCH_d"] = stoch.stoch_signal()
        
        # Get current market conditions
        current_rsi = float(data["RSI"].iloc[-1])
        current_macd = float(data["MACD"].iloc[-1])
        current_macd_signal = float(data["MACD_signal"].iloc[-1])
        current_stoch_k = float(data["STOCH_k"].iloc[-1])
        current_stoch_d = float(data["STOCH_d"].iloc[-1])
        
        # Store initial price when starting a new strategy
        global initial_price, current_strategy
        if current_strategy is None or strategy_name != current_strategy:
            initial_price = current_price
            current_strategy = strategy_name
            logger.info(f"Starting new strategy {strategy_name} at price {initial_price}")
        
        # Always update initial price if not set
        if initial_price is None:
            initial_price = current_price
            logger.info(f"Setting initial price to: ${initial_price:,.2f}")
        
        # Calculate performance based on initial price
        performance = ((current_price - initial_price) / initial_price) * 100
        logger.info(f"Performance calculation: ({current_price} - {initial_price}) / {initial_price} * 100 = {performance}")
        
        # Log market analysis
        logger.info(f"\n{'='*50}")
        logger.info(f"Market Analysis for {symbol}:")
        logger.info(f"{'='*50}")
        logger.info(f"RSI: {current_rsi:.2f}")
        logger.info(f"MACD: {current_macd:.2f} vs Signal: {current_macd_signal:.2f}")
        logger.info(f"Stochastic K: {current_stoch_k:.2f} D: {current_stoch_d:.2f}")
        
        # Evaluate all available strategies
        strategies = []
        
        # RSI Strategy
        if current_rsi < 30:
            strategies.append({"name": "RSI Oversold", "action": "BUY", "confidence": 0.8})
            logger.info("RSI Strategy: Oversold condition detected")
        elif current_rsi > 70:
            strategies.append({"name": "RSI Overbought", "action": "SELL", "confidence": 0.8})
            logger.info("RSI Strategy: Overbought condition detected")
        
        # MACD Strategy
        if current_macd > current_macd_signal:
            strategies.append({"name": "MACD Crossover", "action": "BUY", "confidence": 0.7})
            logger.info("MACD Strategy: Bullish crossover detected")
        elif current_macd < current_macd_signal:
            strategies.append({"name": "MACD Crossover", "action": "SELL", "confidence": 0.7})
            logger.info("MACD Strategy: Bearish crossover detected")
        
        # Stochastic Strategy
        if current_stoch_k < 20 and current_stoch_d < 20:
            strategies.append({"name": "Stochastic Oversold", "action": "BUY", "confidence": 0.6})
            logger.info("Stochastic Strategy: Oversold condition detected")
        elif current_stoch_k > 80 and current_stoch_d > 80:
            strategies.append({"name": "Stochastic Overbought", "action": "SELL", "confidence": 0.6})
            logger.info("Stochastic Strategy: Overbought condition detected")
        
        # Moving Average Strategy
        if data["SMA_20"].iloc[-1] > data["SMA_50"].iloc[-1]:
            strategies.append({"name": "Moving Average Crossover", "action": "BUY", "confidence": 0.5})
            logger.info("MA Strategy: Bullish crossover detected")
        elif data["SMA_20"].iloc[-1] < data["SMA_50"].iloc[-1]:
            strategies.append({"name": "Moving Average Crossover", "action": "SELL", "confidence": 0.5})
            logger.info("MA Strategy: Bearish crossover detected")
        
        # Bollinger Bands Strategy
        if data["close"].iloc[-1] < data["BB_lower"].iloc[-1]:
            strategies.append({"name": "Bollinger Bands", "action": "BUY", "confidence": 0.6})
            logger.info("BB Strategy: Price below lower band")
        elif data["close"].iloc[-1] > data["BB_upper"].iloc[-1]:
            strategies.append({"name": "Bollinger Bands", "action": "SELL", "confidence": 0.6})
            logger.info("BB Strategy: Price above upper band")
        
        # Candlestick Patterns
        patterns = detect_candlestick_patterns(data)
        for pattern_name, signal_strength in patterns:
            action = "BUY" if signal_strength > 0 else "SELL"
            confidence = abs(signal_strength) * 0.5  # Scale pattern confidence
            strategies.append({"name": f"Pattern: {pattern_name}", "action": action, "confidence": confidence})
            logger.info(f"Pattern Strategy: {pattern_name} suggesting {action}")
        
        # Calculate weighted strategy effectiveness
        if strategies:
            buy_confidence = sum(s["confidence"] for s in strategies if s["action"] == "BUY")
            sell_confidence = sum(s["confidence"] for s in strategies if s["action"] == "SELL")
            
            logger.info(f"Strategy confidence scores:")
            logger.info(f"- Buy confidence: {buy_confidence:.2f}")
            logger.info(f"- Sell confidence: {sell_confidence:.2f}")
            
            # Get trade settings from request data
            trade_data = request.get_json() if request else {}
            trade_size_value = float(trade_data.get('size_value', 0))
            trade_size_type = trade_data.get('size_type', 'fixed')
            logger.info(f"Trade Configuration - Size: {trade_size_value} USDT, Type: {trade_size_type}")
            
            # Calculate trade volume in BTC
            trade_volume = trade_size_value / current_price if current_price > 0 else 0
            logger.info(f"Trade volume calculation: {trade_size_value} USDT / ${current_price:,.2f} = {trade_volume:.8f} BTC")
            
            # Check minimum trade requirements
            min_trade_btc = 0.0001
            min_trade_usdt = min_trade_btc * current_price
            logger.info(f"Minimum trade requirements:")
            logger.info(f"- Minimum BTC: {min_trade_btc}")
            logger.info(f"- Minimum USDT: ${min_trade_usdt:,.2f}")
            
            if trade_volume < min_trade_btc:
                error_msg = f"Trade volume {trade_volume:.8f} BTC (${trade_size_value:.2f} USDT) is below minimum requirement of {min_trade_btc} BTC (${min_trade_usdt:,.2f} USDT)"
                logger.error(error_msg)
                return 0.0, "HOLD", "Invalid Trade Size", error_msg
            
            # Get current balances
            balance_data = get_account_balance()
            usdt_balance = float(balance_data.get('USDT', '0.0'))
            btc_balance = float(balance_data.get('XXBT', '0.0'))
            
            logger.info(f"Current balances:")
            logger.info(f"- USDT: ${usdt_balance:,.2f}")
            logger.info(f"- BTC: {btc_balance:.8f}")
            
            action = None
            strategy_name = None
            strategy_desc = None
            
            if buy_confidence > sell_confidence and buy_confidence >= 1.2:
                action = "BUY"
                strategy_name = "Multi-Signal Buy Strategy"
                strategy_desc = "Multiple technical indicators suggest buying"
                
                # Check if we have enough USDT for the trade
                if usdt_balance >= trade_size_value:
                    try:
                        logger.info(f"Executing BUY order: {trade_volume:.8f} BTC at market price")
                        trade_result = execute_kraken_trade(
                            symbol="XBTUSDT",
                            action="buy",
                            volume=trade_volume,
                            ordertype="market"
                        )
                        logger.info(f"Buy trade executed: {trade_result}")
                    except Exception as e:
                        logger.error(f"Error executing buy trade: {e}")
                else:
                    logger.error(f"Insufficient USDT balance. Required: ${trade_size_value:,.2f}, Available: ${usdt_balance:,.2f}")
                    
            elif sell_confidence > buy_confidence and sell_confidence >= 1.2:
                action = "SELL"
                strategy_name = "Multi-Signal Sell Strategy"
                strategy_desc = "Multiple technical indicators suggest selling"
                
                # If we don't have enough BTC, try to buy it first
                if btc_balance < trade_volume:
                    if usdt_balance >= trade_size_value:
                        try:
                            # Add a small buffer (0.1%) to account for fees and rounding
                            adjusted_volume = trade_volume * 1.001
                            logger.info(f"Buying BTC before sell: {adjusted_volume:.8f} BTC")
                            buy_result = execute_kraken_trade(
                                symbol="XBTUSDT",
                                action="buy",
                                volume=adjusted_volume,
                                ordertype="market"
                            )
                            logger.info(f"Initial BTC purchase executed: {buy_result}")
                            
                            # Wait a short time for the order to settle
                            time.sleep(2)
                            
                            # Update BTC balance after purchase
                            balance_data = get_account_balance()
                            btc_balance = float(balance_data.get('XXBT', '0.0'))
                            logger.info(f"Updated BTC balance after purchase: {btc_balance:.8f} BTC")
                        except Exception as e:
                            logger.error(f"Error buying initial BTC: {e}")
                    else:
                        logger.error(f"Insufficient USDT for initial BTC purchase. Required: ${trade_size_value:,.2f}, Available: ${usdt_balance:,.2f}")
                
                # Now execute the sell if we have enough BTC
                if btc_balance >= trade_volume:
                    try:
                        # Use actual available balance for sell, but not more than intended
                        sell_volume = min(btc_balance, trade_volume)
                        logger.info(f"Executing SELL order: {sell_volume:.8f} BTC at market price")
                        trade_result = execute_kraken_trade(
                            symbol="XBTUSDT",
                            action="sell",
                            volume=sell_volume,
                            ordertype="market"
                        )
                        logger.info(f"Sell trade executed: {trade_result}")
                    except Exception as e:
                        logger.error(f"Error executing sell trade: {e}")
                else:
                    logger.error(f"Insufficient BTC balance after purchase. Required: {trade_volume:.8f} BTC, Available: {btc_balance:.8f} BTC")
            else:
                action = "HOLD"
                strategy_name = "Neutral Strategy"
                strategy_desc = "Market conditions unclear, maintaining position"
        else:
            action = "HOLD"
            strategy_name = "No Clear Strategy"
            strategy_desc = "No clear trading signals detected"
        
        # Update market data queue with analysis results
        market_data_queue.put({
            'current_price': f"${current_price:,.2f}",
            'initial_price': f"${initial_price:,.2f}",  # Always show initial price
            'performance': f"{performance:+.2f}%",
            'action': action,
            'strategy': strategy_name,
            'description': strategy_desc,
            'rsi': f"{current_rsi:.2f}",
            'macd': f"{current_macd:.2f}",
            'stochastic': f"{current_stoch_k:.2f}/{current_stoch_d:.2f}",
            'patterns': ', '.join([s["name"] for s in strategies])
        })
        
        logger.info(f"Strategy evaluation complete:")
        logger.info(f"Action: {action}")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Description: {strategy_desc}")
        
        return performance, action, strategy_name, strategy_desc
        
    except Exception as e:
        logger.error(f"Error evaluating strategy performance: {e}")
        return 0.0, "HOLD", "Error", str(e)

def initialize():
    """Initialize Kraken API clients with XBTUSD symbol."""
    # Define parameters
    symbol = "XBTUSD"  # Kraken's Bitcoin/USD pair
    interval = 60  # 1-hour candles
    since = 1622505600  # Example UNIX timestamp

    # Fetch historical data from Kraken
    historical_data = fetch_historical_data(symbol, interval, since)

    # Evaluate strategy
    if historical_data is not None:
        performance, action, strategy_name, strategy_desc = evaluate_strategy_performance("SMA Crossover", historical_data, symbol)
        print(f"Performance: {performance:.2f}%, Recommended Action: {action}")
        print(f"Strategy: {strategy_name} - {strategy_desc}")
    else:
        print("Failed to retrieve historical data.")