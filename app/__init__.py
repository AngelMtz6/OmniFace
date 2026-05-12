from flask import Flask
from .database import init_db


def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.secret_key = 'omniface-hackatec-2026'

    init_db()

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app
