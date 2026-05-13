from flask import Flask
from .database import init_db


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.secret_key = "omniface-2026-secret"

    init_db()

    from .routes import web_bp
    app.register_blueprint(web_bp)

    return app
