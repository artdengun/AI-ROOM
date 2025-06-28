from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import os

from app.controllers.predict_controller import predict_bp
from app.controllers.upload_controller import upload_bp
from app.controllers.ui_controller import ui_bp
from app.models.review_model import db

load_dotenv()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
CORS(app)

# Register blueprint
app.register_blueprint(predict_bp, url_prefix="/api")
app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(ui_bp, url_prefix="/")