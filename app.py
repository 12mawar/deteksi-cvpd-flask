from flask import Flask, render_template, request, redirect, url_for, flash, Response, session, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import mysql.connector
from datetime import datetime
from functools import wraps
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
import base64
from io import BytesIO


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# ===================== LOAD MODEL =====================
model = load_model('model/model_cvpd_segmented1fiks.h5')

# ===================== SEGMENTASI =====================
def segment_yellow_vein(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return cv2.bitwise_and(img, img, mask=mask)

# ===================== DB CONNECTION =====================
def get_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cvpd'
    )

# ===================== SIMPAN HASIL =====================
def simpan_ke_database(nama_file, mimetype, original_data, segmented_data, prediksi, confidence):
    try:
        conn = get_db()
        cursor = conn.cursor()
        username = session.get('user')
        cursor.execute("""
            INSERT INTO hasil_deteksi_cvpd
            (nama_file, mimetype, original_image, segmented_image, prediksi, confidence, waktu, username)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            nama_file,
            mimetype,
            original_data,
            segmented_data,
            prediksi,
            confidence,
            datetime.now(),
            username
        ))
        conn.commit()
        last_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return last_id
    except Exception as e:
        print("Error simpan ke database:", e)
        return None

# ===================== AUTH DECORATOR =====================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session or session.get('role') != 'admin':
            flash('Akses admin ditolak', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ===================== LOGIN =====================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, role FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            session['user'] = user[0]
            session['role'] = user[1]
            if user[1] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard'))
        else:
            flash('Login gagal', 'danger')
    return render_template('login.html')

# ===================== REGISTER =====================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (%s,%s,'user')",
                (username, password)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return redirect(url_for('login'))
        except:
            flash('Username sudah digunakan')
    return render_template('register.html')

# ===================== ADMIN DASHBOARD =====================
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) AS total FROM users")
    total_user = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) AS total FROM hasil_deteksi_cvpd")
    total_deteksi = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) AS total FROM hasil_deteksi_cvpd WHERE prediksi='Sehat'")
    total_sehat = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) AS total FROM hasil_deteksi_cvpd WHERE prediksi='Terinfeksi CVPD'")
    total_terinfeksi = cursor.fetchone()['total']

    cursor.execute("""
        SELECT id, prediksi, confidence, waktu, username
        FROM hasil_deteksi_cvpd
        ORDER BY waktu DESC
    """)
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('admin_dashboard.html',
                           total_user=total_user,
                           total_deteksi=total_deteksi,
                           total_sehat=total_sehat,
                           total_terinfeksi=total_terinfeksi,
                           data=data)

# ===================== LOGOUT =====================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ===================== DASHBOARD USER =====================
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# ===================== DETEKSI =====================
@app.route('/deteksi')
@login_required
def deteksi():
    return render_template('deteksi.html')

# ===================== UPLOAD =====================
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files.get('gambar')
    if not file or file.filename == '':
        flash('File tidak valid')
        return redirect(url_for('deteksi'))

    nama_file = secure_filename(file.filename)
    mimetype = file.mimetype

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))

    segmented_img = segment_yellow_vein(img_resized)

    img_array = segmented_img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array)[0][0])
    label = 'Terinfeksi CVPD' if prediction > 0.9 else 'Sehat'
    confidence = round(prediction * 100 if prediction > 0.9 else (1 - prediction) * 100, 2)

    _, original_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    _, segmented_encoded = cv2.imencode('.jpg', cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

    last_id = simpan_ke_database(
        nama_file,
        mimetype,
        original_encoded.tobytes(),
        segmented_encoded.tobytes(),
        label,
        confidence
    )

    return redirect(url_for('hasil', id=last_id))

# ===================== KAMERA DETEKSI (AJAX) =====================
@app.route('/kamera-deteksi', methods=['POST'])
@login_required
def kamera_deteksi():
    try:
        data_url = request.form.get('image')
        if not data_url:
            return jsonify({'error': 'Data gambar tidak ditemukan'}), 400

        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)

        img = Image.open(BytesIO(image_data)).convert("RGB")
        img = img.resize((150, 150))
        img_np = np.array(img)

        # === SEGMENTASI ===
        segmented_img = segment_yellow_vein(img_np)

        img_array = segmented_img / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = float(model.predict(img_array)[0][0])
        label = 'Terinfeksi CVPD' if prediction > 0.9 else 'Sehat'
        confidence = round(prediction * 100 if prediction > 0.9 else (1 - prediction) * 100, 2)

        # === SIMPAN KE DATABASE ===
        _, original_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        _, segmented_encoded = cv2.imencode('.jpg', cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

        last_id = simpan_ke_database(
            'kamera.jpg',
            'image/jpeg',
            original_encoded.tobytes(),
            segmented_encoded.tobytes(),
            label,
            confidence
        )

        return jsonify({
            'label': label,
            'confidence': confidence,
            'id': last_id
        })

    except Exception as e:
        print("ERROR KAMERA:", e)
        return jsonify({'error': 'Gagal memproses gambar kamera'}), 500


# ===================== HASIL =====================
@app.route('/hasil/<int:id>')
@login_required
def hasil(id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    if session.get('role') == 'admin':
        cursor.execute("SELECT prediksi, confidence FROM hasil_deteksi_cvpd WHERE id=%s", (id,))
    else:
        cursor.execute("SELECT prediksi, confidence FROM hasil_deteksi_cvpd WHERE id=%s AND username=%s",
                       (id, session['user']))

    data = cursor.fetchone()
    cursor.close()
    conn.close()

    if not data:
        return 'Akses ditolak', 403

    return render_template('result.html', label=data['prediksi'], confidence=data['confidence'], id=id)

# ===================== GAMBAR =====================
@app.route('/gambar/<int:id>/<string:jenis>')
@login_required
def gambar(id, jenis):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    if session.get('role') == 'admin':
        cursor.execute("SELECT original_image, segmented_image, mimetype FROM hasil_deteksi_cvpd WHERE id=%s", (id,))
    else:
        cursor.execute("""
            SELECT original_image, segmented_image, mimetype
            FROM hasil_deteksi_cvpd
            WHERE id=%s AND username=%s
        """, (id, session['user']))

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return 'Akses ditolak', 403

    if jenis == 'original':
        return Response(row['original_image'], mimetype=row['mimetype'])
    elif jenis == 'segmented':
        return Response(row['segmented_image'], mimetype=row['mimetype'])
    else:
        return 'Jenis tidak valid', 400
    
# ===================== STATISTIK =====================
@app.route('/statistik')
@login_required
def statistik():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # ================= RINGKASAN =================
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM hasil_deteksi_cvpd
        WHERE username=%s
    """, (session['user'],))
    total_deteksi = cursor.fetchone()['total']

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM hasil_deteksi_cvpd
        WHERE prediksi='Sehat' AND username=%s
    """, (session['user'],))
    total_sehat = cursor.fetchone()['total']

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM hasil_deteksi_cvpd
        WHERE prediksi='Terinfeksi CVPD' AND username=%s
    """, (session['user'],))
    total_terinfeksi = cursor.fetchone()['total']

    # ================= RIWAYAT =================
    cursor.execute("""
        SELECT id, prediksi, confidence, waktu
        FROM hasil_deteksi_cvpd
        WHERE username=%s
        ORDER BY waktu DESC
    """, (session['user'],))
    riwayat = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        'statistik.html',
        total_deteksi=total_deteksi,
        total_sehat=total_sehat,
        total_terinfeksi=total_terinfeksi,
        riwayat=riwayat
    )

# ===================== ERROR HANDLER =====================
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    flash('Ukuran gambar terlalu besar. Maks 5MB.', 'danger')
    return redirect(url_for('deteksi'))

# ===================== ABOUT =====================
@app.route('/about')
@login_required
def about():
    return render_template('tentang.html')

if __name__ == '__main__':
    app.run(debug=True)