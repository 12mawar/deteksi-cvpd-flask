from flask import Flask, render_template, request, redirect, url_for, flash, Response, session, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import mysql.connector
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np

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
    conn = get_db()
    cursor = conn.cursor()
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
        session['user']
    ))
    conn.commit()
    cursor.close()
    conn.close()

# ===================== LOGIN =====================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            session['user'] = username
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
                "INSERT INTO users (username, password) VALUES (%s,%s)",
                (username, password)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return redirect(url_for('login'))
        except:
            flash('Username sudah digunakan')
    return render_template('register.html')

# ===================== LOGOUT =====================
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ===================== DASHBOARD =====================
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# ===================== DETEKSI =====================
@app.route('/deteksi')
def deteksi():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('deteksi.html')

@app.route('/kamera-deteksi', methods=['POST'])
def kamera_deteksi():
    # ⛔ Jangan redirect untuk fetch
    if 'user' not in session:
        return jsonify({'error': 'unauthorized'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'file tidak ditemukan'}), 400

    file = request.files['image']
    img = Image.open(file).convert("RGB")

    # Preprocessing (samakan dengan training)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi CNN
    prediction = model.predict(img_array)[0][0]
    label = "Terinfeksi CVPD" if prediction > 0.9 else "Sehat"
    confidence = round(float(prediction * 100), 2)

    return jsonify({
        'label': label,
        'confidence': confidence
    })


@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

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
    label = "Terinfeksi CVPD" if prediction > 0.9 else "Sehat"
    if prediction > 0.9:
        confidence = round(prediction * 100, 2)
    else:
        confidence = round((1 - prediction) * 100, 2)
    confidence = float(confidence)

    _, original_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    _, segmented_encoded = cv2.imencode('.jpg', cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

    simpan_ke_database(
        nama_file,
        mimetype,
        original_encoded.tobytes(),
        segmented_encoded.tobytes(),
        label,
        confidence
    )

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM hasil_deteksi_cvpd")
    last_id = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    return redirect(url_for('hasil', id=last_id))

# ===================== HASIL =====================
@app.route('/hasil/<int:id>')
def hasil(id):
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prediksi, confidence FROM hasil_deteksi_cvpd
        WHERE id=%s AND username=%s
    """, (id, session['user']))
    data = cursor.fetchone()
    cursor.close()
    conn.close()

    if not data:
        return "Data tidak ditemukan", 404

    return render_template('result.html', label=data[0], confidence=data[1], id=id)

# ===================== GAMBAR =====================
@app.route('/gambar/<int:id>/<string:jenis>')
def gambar(id, jenis):
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()

    if jenis == 'original':
        cursor.execute(
            "SELECT original_image, mimetype FROM hasil_deteksi_cvpd WHERE id=%s",
            (id,)
        )
    elif jenis == 'segmented':
        cursor.execute(
            "SELECT segmented_image, mimetype FROM hasil_deteksi_cvpd WHERE id=%s",
            (id,)
        )
    else:
        return "Jenis gambar tidak valid", 400

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row and row[0]:
        return Response(row[0], mimetype=row[1])
    else:
        return "Gambar tidak ditemukan", 404


@app.route('/riwayat')
def riwayat():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, prediksi, confidence, waktu
        FROM hasil_deteksi_cvpd
        WHERE username = %s
        ORDER BY waktu DESC
    """, (session['user'],))

    data = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('riwayat.html', data=data)

@app.route('/statistik')
def statistik():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # ================= RIWAYAT =================
    cursor.execute("""
        SELECT
            id,
            prediksi,
            confidence,
            waktu
        FROM hasil_deteksi_cvpd
        WHERE username = %s
        ORDER BY waktu DESC
    """, (session['user'],))
    hasil = cursor.fetchall()

    # ================= TOTAL SEHAT =================
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM hasil_deteksi_cvpd
        WHERE prediksi = 'Sehat' AND username = %s
    """, (session['user'],))
    total_sehat = cursor.fetchone()['total']

    # ================= TOTAL TERINFEKSI =================
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM hasil_deteksi_cvpd
        WHERE prediksi = 'Terinfeksi CVPD' AND username = %s
    """, (session['user'],))
    total_terinfeksi = cursor.fetchone()['total']

    # ================= TOTAL DETEKSI =================
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM hasil_deteksi_cvpd
        WHERE username = %s
    """, (session['user'],))
    total_deteksi = cursor.fetchone()['total']

    cursor.close()
    conn.close()

    return render_template(
        'statistik.html',
        hasil=hasil,
        total_sehat=total_sehat,
        total_terinfeksi=total_terinfeksi,
        total_deteksi=total_deteksi
    )


from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    flash('Ukuran gambar terlalu besar. Maks 5MB.', 'danger')
    return redirect(url_for('deteksi'))

# ===================== ABOUT =====================
@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('tentang.html')

if __name__ == '__main__':
    app.run(debug=True)
