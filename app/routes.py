from flask import (Blueprint, render_template, request, redirect,
                   url_for, session, flash, jsonify)
from functools import wraps
from datetime import datetime, date
import base64

from .database import (
    get_all_identities, get_identity_by_account, get_access_logs,
    get_stats, get_identity_full, save_identity_full, get_connection
)
from .auth import create_account, login as auth_login, request_password_reset

web_bp = Blueprint("web", __name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def calc_age(fecha_nac_str: str) -> str:
    try:
        bd    = datetime.strptime(fecha_nac_str, "%Y-%m-%d").date()
        today = date.today()
        age   = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
        return str(age)
    except Exception:
        return "—"


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "account_id" not in session:
            flash("Inicia sesión para continuar", "warning")
            return redirect(url_for("web.login_page"))
        return f(*args, **kwargs)
    return decorated


def needs_renewal(identity: dict | None) -> bool:
    """True si el registro facial tiene más de 365 días o no existe."""
    if not identity:
        return True
    renewal_str = identity.get("last_renewal") or ""
    if not renewal_str:
        return True
    try:
        renewal_date = datetime.fromisoformat(renewal_str.split(".")[0])
        return (datetime.now() - renewal_date).days > 365
    except Exception:
        return True


# ── Rutas públicas ────────────────────────────────────────────────────────────

@web_bp.route("/")
def index():
    if "account_id" in session:
        return redirect(url_for("web.dashboard"))
    return redirect(url_for("web.login_page"))


@web_bp.route("/login", methods=["GET", "POST"])
def login_page():
    if "account_id" in session:
        return redirect(url_for("web.dashboard"))

    if request.method == "POST":
        correo   = request.form.get("correo", "").strip()
        password = request.form.get("password", "")
        account  = auth_login(correo, password)

        if account:
            session["account_id"] = account["id"]
            session["role"]       = account["role"]
            session["name"]       = f"{account['nombre']} {account['ap_paterno']}"
            return redirect(url_for("web.dashboard"))
        flash("Correo o contraseña incorrectos", "danger")

    return render_template("login.html")


@web_bp.route("/register", methods=["GET", "POST"])
def register_page():
    if "account_id" in session:
        return redirect(url_for("web.dashboard"))

    error = None
    if request.method == "POST":
        nombre   = request.form.get("nombre", "").strip()
        ap_pat   = request.form.get("ap_paterno", "").strip()
        ap_mat   = request.form.get("ap_materno", "").strip()
        curp     = request.form.get("curp", "").strip().upper()
        fecha    = request.form.get("fecha_nac", "").strip()
        correo   = request.form.get("correo", "").strip().lower()
        password = request.form.get("password", "")
        pwd2     = request.form.get("password2", "")

        if not all([nombre, ap_pat, curp, fecha, correo, password]):
            error = "Completa todos los campos obligatorios."
        elif len(curp) != 18:
            error = "El CURP debe tener exactamente 18 caracteres."
        elif password != pwd2:
            error = "Las contraseñas no coinciden."
        elif len(password) < 6:
            error = "La contraseña debe tener al menos 6 caracteres."
        else:
            try:
                datetime.strptime(fecha, "%Y-%m-%d")
            except ValueError:
                error = "Fecha inválida. Usa el formato AAAA-MM-DD."

        if not error:
            try:
                create_account(nombre, ap_pat, ap_mat, curp, fecha, correo, password)
                flash("Cuenta creada. Inicia sesión.", "success")
                return redirect(url_for("web.login_page"))
            except ValueError as e:
                error = str(e)

    return render_template("register.html", error=error)


@web_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    sent = False
    if request.method == "POST":
        correo = request.form.get("correo", "").strip()
        ok = request_password_reset(correo)
        if ok:
            sent = True
        else:
            flash("No se encontró ninguna cuenta con ese correo.", "danger")
    return render_template("forgot_password.html", sent=sent)


@web_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("web.login_page"))


# ── Rutas protegidas ──────────────────────────────────────────────────────────

@web_bp.route("/dashboard")
@login_required
def dashboard():
    from .database import get_connection
    conn = get_connection()
    account = dict(conn.execute(
        "SELECT * FROM accounts WHERE id = ?", (session["account_id"],)
    ).fetchone())
    conn.close()

    identity     = get_identity_by_account(session["account_id"])
    renewal_due  = needs_renewal(identity)
    age          = calc_age(account.get("fecha_nac", ""))

    # Últimas detecciones del usuario
    logs = []
    if identity:
        logs = get_access_logs(limit=10, identity_id=identity["id"])

    stats = get_stats()

    return render_template("dashboard.html",
                           account=account,
                           identity=identity,
                           renewal_due=renewal_due,
                           age=age,
                           logs=logs,
                           stats=stats)


@web_bp.route("/profile")
@login_required
def profile():
    from .database import get_connection
    conn = get_connection()
    account = dict(conn.execute(
        "SELECT * FROM accounts WHERE id = ?", (session["account_id"],)
    ).fetchone())
    conn.close()

    identity     = get_identity_by_account(session["account_id"])
    age          = calc_age(account.get("fecha_nac", ""))
    renewal_due  = needs_renewal(identity)

    return render_template("profile.html", account=account,
                           identity=identity, age=age,
                           renewal_due=renewal_due)


@web_bp.route("/my-history")
@login_required
def my_history():
    identity = get_identity_by_account(session["account_id"])
    logs     = []
    if identity:
        logs = get_access_logs(limit=50, identity_id=identity["id"])
    return render_template("my_history.html", logs=logs,
                           identity=identity)

@web_bp.route("/register_face")
@login_required
def register_face_page():
    return render_template("register_face.html")

@web_bp.route("/api/register_face", methods=["POST"])
@login_required
def api_register_face():
    data = request.get_json()
    if not data or "frames" not in data:
        return jsonify({"success": False, "error": "No frames provided"}), 400
        
    frames_b64 = data["frames"]
    if not frames_b64:
        return jsonify({"success": False, "error": "Empty frames array"}), 400
        
    face_blobs = []
    for b64 in frames_b64:
        try:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            face_blobs.append(img_bytes)
        except Exception:
            pass
            
    if not face_blobs:
         return jsonify({"success": False, "error": "Could not decode any frame"}), 400
         
    conn = get_connection()
    acc = dict(conn.execute("SELECT * FROM accounts WHERE id = ?", (session["account_id"],)).fetchone())
    conn.close()
    
    save_identity_full(
        nombre=acc['nombre'],
        ap_paterno=acc['ap_paterno'],
        ap_materno=acc['ap_materno'],
        curp=acc['curp'],
        fecha_nac=acc['fecha_nac'],
        correo=acc['correo'],
        face_blobs=face_blobs,
        account_id=acc['id']
    )
    
    flash("Registro facial completado exitosamente. La aplicación de escritorio procesará las muestras la próxima vez que se inicie.", "success")
    return jsonify({"success": True})
