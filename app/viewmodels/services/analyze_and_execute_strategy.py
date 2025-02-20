from flask import Flask,Blueprint, request, jsonify
from trading_logic import evaluate_strategy_performance
from market_data import fetch_historical_data
import logging
from routes import routes

logger = logging.getLogger(__name__)
# Crear instancia de Flask 
# app = Flask(__name__)
analyze_and_execute_strategy_bp = Blueprint('analyze_and_execute_strategy', __name__)

@analyze_and_execute_strategy_bp.route('/analyze_and_execute_strategy', methods=['POST'])
def analyze_and_execute_strategy():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        symbol = data.get('symbol', 'XBTUSDT')
        historical_data = fetch_historical_data(symbol)
        if historical_data is None or historical_data.empty:
            return jsonify({"status": "error", "message": "Could not fetch historical data"}), 400

        performance, action, strategy_name, strategy_desc = evaluate_strategy_performance(
            "Dynamic Strategy", historical_data, symbol, data.get('use_custom_strategies', False))

        return jsonify({
            "status": "success",
            "performance": performance,
            "action": action,
            "strategy": strategy_name,
            "description": strategy_desc
        })
    except Exception as e:
        logger.error(f"Error in analyze_and_execute_strategy: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def initialize_app(): 
    logger.info("App initialized")