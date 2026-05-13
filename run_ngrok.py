"""
run_ngrok.py — Levanta OmniFace con túnel HTTPS de ngrok.

Uso:
    python run_ngrok.py                  # usa el token embebido
    python run_ngrok.py --token <TOKEN>  # sobreescribe con otro token

La URL pública HTTPS se imprime en consola. Ábrela en el navegador para
acceder a la cámara sin que el navegador bloquee getUserMedia().
"""
import argparse

from app import create_app

PORT = 5000

# Token de ngrok — cuenta verificada
_NGROK_TOKEN = "3DfpS4y06kmUv9b9uL88neDp3gM_7UoEvsJTjGYRHWpzFDJWm"


def start_ngrok(token: str | None = None) -> str:
    from pyngrok import ngrok

    # Siempre aplicar el token (embebido o el que se pase por CLI)
    ngrok.set_auth_token(token or _NGROK_TOKEN)

    # Abre el túnel HTTP→HTTPS
    tunnel = ngrok.connect(PORT, "http")
    public_url: str = tunnel.public_url

    # ngrok a veces devuelve http:// aunque el túnel sea HTTPS; forzar https://
    if public_url.startswith("http://"):
        public_url = "https://" + public_url[7:]

    return public_url


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniFace + ngrok HTTPS")
    parser.add_argument("--token", default=None,
                        help="ngrok authtoken (opcional, extiende la sesión)")
    args = parser.parse_args()

    print("=" * 60)
    print("  OmniFace — iniciando túnel HTTPS con ngrok...")
    print("=" * 60)

    try:
        public_url = start_ngrok(args.token)
        print(f"\n  ✅  URL pública HTTPS:")
        print(f"      {public_url}\n")
        print("  Abre esa URL en cualquier navegador o dispositivo.")
        print("  La cámara funcionará porque es HTTPS (contexto seguro).")
        print("  Ctrl+C para detener.\n")
        print("=" * 60)
    except Exception as exc:
        print(f"\n  ⚠️  No se pudo iniciar ngrok: {exc}")
        print("  Asegúrate de tener conexión a Internet.")
        print("  La app seguirá corriendo solo en localhost:5000\n")
        print("=" * 60)

    # Arranca Flask en el hilo principal (sin reloader para evitar conflictos)
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=PORT,
            threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
