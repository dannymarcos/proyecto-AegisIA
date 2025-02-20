from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from app.models.create_db import db

class Application:
   
    def __init__(self, config_object):
        self.app = Flask(__name__, template_folder="iu/templates", static_folder="iu/static")
        self.app.config.from_object(config_object)
        db.init_app(self.app) #Inicializamos db dentro del constructor
        CORS(self.app)
        
        self.init_db()

    def init_db(self):
        with self.app.app_context():
           db.create_all()

    def register_blueprint(self, blueprint):
        self.app.register_blueprint(blueprint)

    def run(self, host='0.0.0.0', port=8080, debug=True, threaded=True):
        self.app.run(host=host, port=port, debug=debug, threaded=threaded)

    def app_context(self): 
        return self.app.app_context()


