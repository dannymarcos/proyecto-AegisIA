#proyecto_aegisia_main/main.py
import logging
import sys
from flask import Flask
from app.Aplicacion import Application
from app.iu.routes import routes_bp
from app.config import Config
# from app.services.kraken_service import KrakenService
# from app.services.trading_service import TradingService


# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Crear la instancia de la aplicación fuera del bloque principal
#app = create_app()
# # Crear instancia de la aplicación

app_instance = Application(Config)


def initialize_server(
    app_instance
    ):
    """Inicializa los componentes del servidor con el manejo adecuado de errores."""
    try:


        logger.info("Iniciando la inicialización de la aplicación...")
        
        with app_instance.app_context():
            logger.info("Inicializando la API de Kraken...")
            #kraken_service = KrakenService()
            #kraken_service.initialize()
            logger.info("API de Kraken inicializada exitosamente")
            
            logger.info("Inicializando el sistema de trading...")
            #trading_service = TradingService()
            #trading_service.initialize()
            logger.info("Sistema de trading inicializado exitosamente")
            
            logger.info("Iniciando el monitor de mercado...")
            # start_market_monitor()  # Asegúrate de que esta función esté bien definida
            logger.info("Monitor de mercado iniciado exitosamente")
            
        return True
    except Exception as e:
        logger.error(f"Error de inicialización: {str(e)}", exc_info=True)
        return False

# Add a test route to verify the server is running
# @app.route('/test')
# def test():
#     return "Server is running!"

if __name__ == "__main__":
    # # Registrar Blueprints si es necesario
    app_instance.register_blueprint(routes_bp)
    try:
        logger.info("Iniciando la inicialización del servidor...")
        
        if not initialize_server(app_instance):
            logger.error("La inicialización del servidor falló. Saliendo.")
            sys.exit(1)
        
        logger.info("Iniciando el servidor de desarrollo de Flask...")
        # Ejecutar el servidor de desarrollo de Flask con threaded=True para un mejor rendimiento
        # app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
        app_instance.run()
        
    except Exception as e:
        logger.error(f"Error al iniciar el servidor: {e}", exc_info=True)
        sys.exit(1)
