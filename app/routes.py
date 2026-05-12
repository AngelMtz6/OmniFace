import base64
import json
import time

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context

from .camera import VideoCamera
from .database import (delete_identity, get_access_logs, get_all_identities,
                        get_stats, save_identity)
from .recognition import RecognitionEngine

main_bp = Blueprint('main', __name__)

camera = VideoCamera()
engine = RecognitionEngine()


# ── Streaming ────────────────────────────────────────────────────────────────

def _generate_frames():
    last_frame_id = -1
    while True:
        frame, frame_id = camera.get_frame()
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.01)  # Esperar un poco por un nuevo frame
            continue

        last_frame_id = frame_id
        processed = engine.process_frame(frame)
        ret, buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        
        # Sincronizar con el framerate deseado (aprox 30 FPS)
        time.sleep(0.02)


@main_bp.route('/video_feed')
def video_feed():
    return Response(_generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Pages ─────────────────────────────────────────────────────────────────────

@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/register')
def register():
    return render_template('register.html')


@main_bp.route('/admin')
def admin():
    return render_template('admin.html')


# ── API ───────────────────────────────────────────────────────────────────────

@main_bp.route('/capture_frame')
def capture_frame():
    frame, _ = camera.get_frame()
    if frame is None:
        return jsonify({'error': 'Cámara no disponible'}), 503
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({'error': 'Error al codificar imagen'}), 500
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{b64}'})


@main_bp.route('/enroll', methods=['POST'])
def enroll():
    data       = request.get_json()
    name       = (data.get('name') or '').strip()
    frames_b64 = data.get('frames', [])

    if not name:
        return jsonify({'error': 'Nombre requerido'}), 400
    if len(frames_b64) < 3:
        return jsonify({'error': 'Se necesitan mínimo 3 capturas'}), 400

    detector   = engine.detector
    face_blobs = []

    for i, b64 in enumerate(frames_b64):
        try:
            img_bytes = base64.b64decode(b64.split(',')[1])
            nparr     = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in rects:
                face    = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                ok, buf = cv2.imencode('.jpg', face)
                if ok:
                    face_blobs.append(buf.tobytes())
                break  # una cara por frame
        except Exception:
            continue

    if not face_blobs:
        return jsonify({'error': 'No se detectó ningún rostro en las capturas'}), 400

    try:
        identity_id = save_identity(name, face_blobs)
        engine.retrain()
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

    return jsonify({'success': True, 'id': identity_id,
                    'name': name, 'samples': len(face_blobs)})


@main_bp.route('/identities')
def identities():
    return jsonify(get_all_identities())


@main_bp.route('/identity/<int:identity_id>', methods=['DELETE'])
def delete(identity_id):
    delete_identity(identity_id)
    engine.retrain()
    return jsonify({'success': True})


@main_bp.route('/logs')
def logs():
    return jsonify(get_access_logs())


@main_bp.route('/stats')
def stats():
    return jsonify(get_stats())


@main_bp.route('/alert_status')
def alert_status():
    return jsonify(engine.alert_manager.get_status())


@main_bp.route('/face_status')
def face_status():
    """Detecta si hay una cara visible en el frame actual. Usado por el enrollamiento guiado."""
    frame, _ = camera.get_frame()
    if frame is None:
        return jsonify({'face_detected': False, 'count': 0})
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = engine.detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return jsonify({'face_detected': len(rects) > 0, 'count': int(len(rects))})


# ── Server-Sent Events (alertas en tiempo real) ───────────────────────────────

@main_bp.route('/events')
def events():
    def _stream():
        last_alert = False
        while True:
            status = engine.alert_manager.get_status()
            alert  = status['alert_active']
            if alert != last_alert:
                yield f'data: {json.dumps({"alert": alert})}\n\n'
                last_alert = alert
            time.sleep(0.5)

    return Response(stream_with_context(_stream()), mimetype='text/event-stream')
