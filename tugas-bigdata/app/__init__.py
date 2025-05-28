from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    os.makedirs('uploads', exist_ok=True)

    from .routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app
