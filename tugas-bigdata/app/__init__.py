from flask import Flask
from app.routes.uploads_routes import upload_bp
from app.routes.pegawai_routes import pegawai_bp
from app.routes.jabatan_routes import jabatan_bp
import os

def create_app():
    app = Flask(__name__)
    os.makedirs('uploads', exist_ok=True)
    app.register_blueprint(upload_bp)
    app.register_blueprint(jabatan_bp)
    app.register_blueprint(pegawai_bp)

    return app
