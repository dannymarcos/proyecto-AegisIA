from app.viewmodels.api.market_data import get_account_balance,execute_kraken_trade

from flask import Blueprint, request, jsonify
import logging
import time
logger = logging.getLogger(__name__)

execute_trade_bp = Blueprint('execute_trade', __name__)

@execute_trade_bp.route('/execute_trade', methods=['POST'])
def execute_trade():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        symbol = data.get('symbol')
        action = data.get('action')
        ordertype = data.get('ordertype', 'market')
        volume = data.get('volume')
        price = data.get('price')
        size_type = data.get('size_type', 'fixed')
        take_profit = data.get('take_profit')
        stop_loss = data.get('stop_loss')
        leverage = data.get('leverage')

        logger.info(f"Received trade request: symbol={symbol}, action={action}, volume={volume}, price={price}")

        if not symbol or not action or not volume:
            return jsonify({"status": "error", "message": "Missing required trade parameters"}), 400

        # Check balance and convert if necessary
        balance = get_account_balance()
        logger.info(f"Current balance: {balance}")

        # Convert price to float if it's not None
        if price is not None:
            try:
                price = float(price)
            except (TypeError, ValueError):
                logger.error(f"Invalid price value: {price}")
                return jsonify({"status": "error", "message": "Invalid price value"}), 400

        # For market orders, get current price from balance check response
        if ordertype == 'market':
            # Use a default price for balance check if needed
            price = 1.0  # This is just for balance check, actual price will be determined by market

        if action == 'buy' and balance.get('USDT'):
            usdt_balance = float(balance['USDT'])
            required_balance = float(volume) * price
            logger.info(f"Buy order - Required USDT: {required_balance}, Available: {usdt_balance}")
            if usdt_balance < required_balance:
                return jsonify({"status": "error", "message": f"Insufficient USDT balance. Required: {required_balance}, Available: {usdt_balance}"}), 400
        elif action == 'sell' and symbol and balance.get(symbol):
            crypto_balance = float(balance[symbol])
            required_volume = float(volume)
            logger.info(f"Sell order - Required {symbol}: {required_volume}, Available: {crypto_balance}")
            if crypto_balance < required_volume:
                return jsonify({"status": "error", "message": f"Insufficient {symbol} balance. Required: {required_volume}, Available: {crypto_balance}"}), 400

        trade_result = execute_kraken_trade(
            symbol=symbol,
            action=action,
            volume=volume,
            ordertype=ordertype,
            price=price,
            leverage=leverage
        )

        logger.info(f"Trade result: {trade_result}")

        if trade_result.get('error'):
            return jsonify({"status": "error", "message": trade_result['error']}), 400

        return jsonify({"status": "success", "trade_result": trade_result})
    except Exception as e:
        logger.error(f"Error in execute_trade: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


class KrakenFuturesAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def get_balance(self):
        pass
    def place_order(self, symbol, side, quantity, price = None, order_type='market'):
        pass

# def initialize(): 
#     logger.info("Kraken API initialized")