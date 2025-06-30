from flask import Flask, render_template
from app.controllers.ui_controller import ui_bp
# from app.routes.uploads_routes import upload_bp
# from app.routes.pegawai_routes import pegawai_bp
# from app.routes.jabatan_routes import jabatan_bp
import os

def create_app():
    app = Flask(__name__,  static_folder='static', template_folder='templates')
    app.secret_key = 'ini_rahasia_banget_12345'
    app.register_blueprint(ui_bp)

    # app.register_blueprint(upload_bp, url_prefix='/upload')
    # app.register_blueprint(jabatan_bp)
    # app.register_blueprint(pegawai_bp)

    return app
