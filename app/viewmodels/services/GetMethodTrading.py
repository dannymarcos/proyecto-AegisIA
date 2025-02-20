# app/viewmodels/services/GetMethodTrading.py
from flask import current_app


class GetMethodTrading:

    def __init__(self):

        self.method = None

    
    def get_method(self):
        # Obtener el modo de trading desde la configuracion. con "spot" como valor predeterminado
        self.method = current_app.config.get("TRADING_MODE", "spot")

        return self.method
