# app/viewmodels/api/futures/KrakenFuturesAPI

import requests
import logging
from flask import jsonify

logger = logging.getLogger(__name__)

class KrakenFuturesAPI:

    def __init__(self):

        self.__endpoint = "https://futures.kraken.com/derivatives/api/v3/tickers"
        self.data = None

    def get_ticker_kraken(self):
        try:
            response = requests.get(self.__endpoint)
            if response.status_code != 200:
                logger.error(f"Error fetching tickers: HTTP {response.status_code}")
                return jsonify({"error": "Failed to fetch tickers"}), 500

            self.data = response.json()
            if "error" in self.data and self.data["error"]:
                logger.error(f"Kraken API error: {data['error']}")
                return {"error": str(self.data["error"])}, 500

        except Exception as e:
            logger.error(f"Error getting cryptocurrencies: {e}")
            return {"error": str(e)}, 500
        
        return self.data, 200

    def get_symbol_and_markPrice(self):

        if not self.data:
            return {"error":"No data available"}, 500
        try:
            tickers = self.data.get("tickers", [])
            cryptos = []
            for  ticker in tickers:
                pair = ticker.get("pair")
                markPrice = ticker.get("markPrice")
                if pair and markPrice:
                    cryptos.append({
                        "symbol":pair,
                        "price":markPrice
                    })
        except Exception as e:
            logger.error(f"Error getting cryptocurrencies: {e}")
            return jsonify({"error": str(e)}), 500
        return {"cryptos":cryptos}, 200