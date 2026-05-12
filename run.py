from app import create_app

app = create_app()

if __name__ == '__main__':
    print("OmniFace iniciando en http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
