from flask import Flask, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy

# Create Flask app instance
app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize SQLAlchemy
db = SQLAlchemy()