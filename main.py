import logging
import sys
from flask import Flask, render_template
from app_init import app, initialize_app
from kraken_api import initialize as kraken_initialize
from trading_logic import initialize as trading_initialize, start_market_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def initialize_server():
    """Initialize server components with proper error handling"""
    try:
        logger.info("Starting application initialization...")
        initialize_app()
        logger.info("App initialized successfully")
        
        logger.info("Initializing Kraken API...")
        kraken_initialize()
        logger.info("Kraken API initialized successfully")
        
        logger.info("Initializing trading system...")
        trading_initialize()
        logger.info("Trading system initialized successfully")
        
        logger.info("Starting market monitor...")
        start_market_monitor()
        logger.info("Market monitor started successfully")
        
        return True
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        return False

# Add a test route to verify the server is running
@app.route('/test')
def test():
    return "Server is running!"

if __name__ == "__main__":
    try:
        logger.info("Starting server initialization...")
        
        if not initialize_server():
            logger.error("Server initialization failed. Exiting.")
            sys.exit(1)
        
        logger.info("Starting Flask development server...")
        # Run Flask development server with threaded=True for better performance
        app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
        
    except Exception as e:
        logger.error(f"Failed to start the server: {e}", exc_info=True)
        sys.exit(1)