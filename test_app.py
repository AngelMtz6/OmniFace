import sys, traceback
try:
    from app import create_app
    app = create_app()
    rules = [str(r) for r in app.url_map.iter_rules()]
    print("OK")
    for r in rules:
        print(" ", r)
except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()
