import pymysql
pymysql.install_as_MySQLdb()

import pandas as pd
import json
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import logging
from datetime import datetime
import os
from model_definition import LSTMModel, load_model, predict_disease 
from data_penyakit import deskripsi_penyakit, gambar_penyakit
from flask import request, redirect, url_for, flash, session
from collections import Counter
import sqlite3
from flask import make_response, render_template


app = Flask(__name__)
app.secret_key = os.urandom(24)  

# Database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flashdb'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

# Load symptoms dataset
def load_gejala():
    try:
        gejala_df = pd.read_csv('Dataset_Gejala_Ayam.csv')
        return gejala_df
    except Exception as e:
        logging.error(f"Error loading symptoms dataset: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Initialize Model
input_size = 36  # Number of symptom features
hidden_size = 64  # Hidden layer size
num_layers = 2  # Number of LSTM layers
num_classes = 9  # Number of disease classes
dropout_rate = 0.2  # Dropout rate

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# Check if model file exists before loading
model_path = "model_lstm_penyakit_ayam.pth"
if not os.path.exists(model_path):
    logging.error(f"Model file not found: {model_path}")
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load model weights - fixed the duplicated line
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Mapping prediction results to disease names
penyakit_mapping = {
    0: "Sehat", 1: "Tetelo", 2: "Feses Kapur", 3: "Gumboro",
    4: "Penyakit Ngorok", 5: "Flu Burung", 6: "Kolibasilosis",
    7: "Snot", 8: "Kolera",
}

deskripsi_penyakit = {
    "Sehat": "Ayam dalam kondisi sehat, tanpa gejala penyakit yang terdeteksi.",
    "Tetelo": "Penyakit tetelo atau Newcastle merupakan penyakit yang disebabkan oleh virus yang sangat berbahaya dan bisa menyebabkan kematian pada unggas, terutama ayam. Penyakit ini disebabkan oleh virus avian paramyxovirus tipe 1 yang ganas. Ayam yang belum divaksin bisa mengalami kematian mendadak, tampak lemas, dan mengalami gangguan pernapasan",
    "Gumboro": "Penyakit gumboro atau bursal merupakan penyakit yang sering menyerang anak ayam di seluruh dunia. Penyakit ini disebabkan oleh virus IBDV. Ayam yang terinfeksi biasanya tampak lemas, mengalami diare encer, bulunya kusut, dan terlihat dehidrasi (kekurangan cairan). Penyakit ini menyebar dengan cepat (morbiditas tinggi), dan meskipun biasanya tidak banyak yang mati, beberapa jenis virus yang ganas bisa menyebabkan kematian hingga lebih dari 60%.",
    "Feses Kapur": "Penyakit fese kapur (Pullorum) atau dikenal sebagai berak kapur merupakan penyakit yang disebabkan oleh Salmonella enterica serotipe Gallinarum biovar Pullorum yang ditularkan secara vertikal dan ditandai dengan angka kematian yang sangat tinggi pada ayam muda dan kalkun. Unggas yang terkena penyakit ini akan berkerumun di dekat sumber panas, terlihat anoreksia, lemah, dan lesu, serta terdapat kotoran berwarna putih yang menempel di area ventilasi.",
    "Penyakit Ngorok": "Penyakit ngorok atau dikenal sebagai chronic respiratory disease adalh penyakit yang menyerang pernapasan secara kronis. Penyakit ngorok disebabkan oleh infeksi bakteri mycoplasma gallisepticum. Penyakit ngorok ini biasanya menyerang organ mulai dari rongga hidung sampai dengan kantong udara, yang menyebabkan napas ayam menjadi berat dan tersumbat sehingga menimbulkan suara seperti ngorok.",
    "Flu Burung": "Peyakit flu burung atau avian influenza merupakan jenis penyakit menular yang bersifat akut pada unggas. Flu burung disebabkan oleh virus influenza tipe A, yang menyerang saluran pernafasan unggas dengan tingkat penularan yang cukup tinggi. Ifeksi virus ini dapat menyebar pada burung, terkadang juga dapat menyebar ke manusia. Infeksi avian influenza ini mampu menimbulkan moralitas tinggi dengan kematian mendadak tanpa adanya gejala yang dialami oleh ayam.",
    "Kolibasilosis":  "Penyakit kolibacillosis merupakan penyakit yang disebabkan oleh infeksi strain Escherichia coli. Sindrom yang terkait dengan kolibasilosis dapat bervariasi dan meliputi septikemia fatal akut, airsacculitis, perikarditis, perihepatitis, peritonitis, dan penipisan limfositik pada bursa dan timus. Diagnosis biasanya dilakukan dengan isolasi kultur murni E coli, yang konsisten dengan kolibasilosis, dari lesi pada unggas. Sebagian besar isolat bakteri resisten terhadap berbagai antimikroba, sehingga pencegahan paparan melalui manajemen yang baik sangat dianjurkan.",
    "Snot": "Snot atau Infectious coryza adalah penyakit pernapasan akut pada ayam yang disebabkan oleh Avibacterium paragallinarum. Tanda-tanda klinis meliputi penurunan aktivitas, keluarnya cairan dari hidung, bersin-bersin, dan pembengkakan pada wajah. Diagnosis dugaan didasarkan pada tanda-tanda klinis yang khas pada ayam yang rentan. Diagnosis dikonfirmasi dengan uji PCR atau kultur bakteri. Pengobatan antimikroba dini dapat membantu pemulihan unggas yang terinfeksi. Pencegahan didasarkan pada praktik manajemen yang baik, termasuk tindakan biosekuriti yang tepat dan vaksinasi dengan serovar yang ada di populasi lokal. ",
    "Kolera": "Kolera unggas adalah penyakit menular pada burung yang disebabkan oleh bakteri Pasteurella multocida. Pada kasus yang parah (akut), penyakit ini bisa menyebabkan kematian dalam jumlah besar. Sedangkan pada kasus yang lebih ringan (kronis), kolera unggas dapat menyebabkan kelumpuhan, pembengkakan pada jengger ayam, gangguan pernapasan seperti pneumonia pada kalkun, atau gerakan kepala yang tidak normal (torticollis). Namun, ada juga burung yang terinfeksi tetapi tidak menunjukkan gejala. Untuk pencegahan, tersedia vaksin hidup yang dilemahkan dan vaksin bakteri yang ditambah bahan penguat (adjuvan). Bakteri penyebab penyakit ini juga masih bisa diobati dengan beberapa jenis antibiotik.",
}

gambar_penyakit = {
    "Tetelo": "Tetello.png",
    "Feses Kapur": "Feses_Kapur.png",
    "Gumboro": "Gumboro.png",
    "Penyakit Ngorok": "Penyakit_Ngorok.png",
    "Flu Burung": "Flu Burung.png",
    "Kolibasilosis": "Kolibasilosis.png",
    "Snot": "Snot.png",
    "Kolera": "kolera.png",
    "Sehat": "Sehat.png",
    "Tidak Ada Gambar Yang Tersedia": "default.png"
}

def get_prediction(gejala):
    # Konversi input gejala ke tensor dengan shape (batch=1, seq_len=1, fitur=36)
    input_vector = torch.tensor(gejala).float().unsqueeze(0).unsqueeze(1)  # shape: (1, 1, 36)

    with torch.no_grad():
        output = model(input_vector)
        _, predicted_class = torch.max(output, 1)
    predicted_disease = penyakit_mapping.get(predicted_class.item(), "Tidak Dikenal")
    disease_info = deskripsi_penyakit.get(predicted_disease, "Deskripsi tidak ditemukan.")
    gambar_penyakit = {
        "Tetelo": "Tetello.png",
        "Feses Kapur": "Feses_Kapur.png",
        "Gumboro": "Gumboro.png",
        "Penyakit Ngorok": "Penyakit_Ngorok.png",
        "Flu Burung": "Flu_Burung.png",
        "Kolibasilosis": "Kolibasilosis.png",
        "Snot": "Snot.png",
        "Kolera": "kolera.png",
        "Sehat": "Sehat.png",
        "Tidak Ada Gambar Yang Tersedia": "default.png"
    }
    gambar = gambar_penyakit.get(predicted_disease, gambar_penyakit["Tidak Ada Gambar Yang Tersedia"])
    return predicted_disease, gambar, disease_info

@app.route('/hasil', methods=['GET', 'POST'])
def hasil_page():
    return render_template('hasil.html')

@app.route("/")
def home():
    username = session.get('username')
    return render_template("index.html", username=username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'loggedin' in session:
        return redirect(url_for('riwayat'))  # Atau halaman utama lain

    next_page = request.args.get('next')

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and user['password'] == password:
            session['loggedin'] = True
            session['username'] = user['username']
            return redirect(next_page or url_for('riwayat'))
        else:
            flash('Login gagal. Email atau password salah.', 'danger')
            return redirect(url_for('login', next=next_page))

    return render_nocache_template('login.html', next=next_page)


@app.route('/register', methods=['GET', 'POST'])
def register():
    next_page = request.args.get('next')  # tangkap next dari URL
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Email sudah digunakan.', 'danger')
            return redirect(url_for('register', next=next_page))

        cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)',
                       (username, email, password))
        mysql.connection.commit()
        cursor.close()

        flash('Registrasi berhasil. Silakan login.', 'success')
        return redirect(url_for('login', next=next_page))  # arahkan ke login dengan next

    return render_template('register.html', next=next_page)


@app.route("/about")
def about():
    username = session.get('username')
    return render_nocache_template("about.html", username=username)

@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if 'loggedin' not in session:
        flash('Harap login terlebih dahulu untuk mengakses halaman deteksi', 'danger')
        return redirect(url_for('login', next=request.path))

    username = session.get('username')

    try:
        cursor = mysql.connection.cursor()
        # Ambil riwayat prediksi user untuk ditampilkan di halaman deteksi
        cursor.execute('SELECT * FROM riwayat_prediksi WHERE username = %s ORDER BY tanggal DESC', (username,))
        history = cursor.fetchall()
        cursor.close()
    except Exception as e:
        flash(f"Terjadi kesalahan saat mengambil riwayat: {str(e)}", "danger")
        history = []

    if request.method == 'POST':
        gejala = []
        for i in range(1, 37):
            val = request.form.get(f'gejala{i}')
            if val is None:
                flash("Silakan pilih semua gejala (Ya/Tidak) sebelum melakukan prediksi", "danger")
                return redirect(url_for('deteksi'))
            gejala.append(int(val))

        # Prediksi penyakit berdasarkan gejala
        prediction, gambar, disease_info = get_prediction(gejala)

        # Simpan hasil prediksi ke database
        try:
            cursor = mysql.connection.cursor()
            # Ambil timestamp saat ini untuk 'tanggal'
            now = datetime.now()
            tanggal = now.strftime('%Y-%m-%d %H:%M:%S')

            # Insert data ke database
            cursor.execute(
                "INSERT INTO riwayat_prediksi (username, tanggal, hasil, detail) VALUES (%s, %s, %s, %s)",
                (username, tanggal, prediction, disease_info)  # disease_info digunakan untuk 'detail'
            )
            mysql.connection.commit()
            cursor.close()
        except Exception as e:
            flash(f"Gagal menyimpan riwayat prediksi: {str(e)}", "danger")

        # Ambil riwayat terbaru
        try:
            cursor = mysql.connection.cursor()
            cursor.execute('SELECT * FROM riwayat_prediksi WHERE username = %s ORDER BY tanggal DESC', (username,))
            history = cursor.fetchall()
            cursor.close()
        except Exception as e:
            flash(f"Terjadi kesalahan saat mengambil riwayat terbaru: {str(e)}", "danger")
            history = []

        return render_template('hasil.html', prediction=prediction, gambar=gambar, disease_info=disease_info, history=history)

    return render_nocache_template('deteksi.html', username=username, history=history)

@app.route("/detail_penyakit")
def detail_penyakit():
    username = session.get('username')
    return render_nocache_template("detail_penyakit.html", username=username)

# Standardize disease page routes
@app.route('/tetelo')
def tetelo():
    
    return render_template('tetelo.html')

@app.route('/feses-kapur')
def feses_kapur():
    return render_template('feses_kapur.html')

@app.route('/gumboro')
def gumboro():
    return render_template('gumboro.html')

@app.route('/penyakit-ngorok')
def penyakit_ngorok():
    return render_template('penyakit_ngorok.html')

@app.route('/flu-burung')
def flu_burung():
    return render_template('flu_burung.html')

@app.route('/kolibasilosis')
def kolibasilosis():
    return render_template('kolibasilosis.html')

@app.route('/snot')
def snot():
    return render_template('snot.html')

@app.route('/kolera')
def kolera():
    return render_template('kolera.html')

@app.route("/pencegahan_penyakit")
def pencegahan_penyakit():
    username = session.get('username')
    return render_nocache_template("pencegahan_penyakit.html", username=username)

# Standardize prevention page routes
@app.route('/pencegahan/tetelo')
def pencegahan_tetelo():
    return render_template('pencegahan_tetelo.html')

@app.route('/pencegahan/feses-kapur')
def pencegahan_feses_kapur():
    return render_template('pencegahan_feses_kapur.html')

@app.route('/pencegahan/gumboro')
def pencegahan_gumboro():
    return render_template('pencegahan_gumboro.html')

@app.route('/pencegahan/penyakit-ngorok')
def pencegahan_penyakit_ngorok():
    return render_template('pencegahan_penyakit_ngorok.html')

@app.route('/pencegahan/flu-burung')
def pencegahan_flu_burung():
    return render_template('pencegahan_flu_burung.html')

@app.route('/pencegahan/kolibasilosis')
def pencegahan_kolibasilosis():
    return render_template('pencegahan_kolibasilosis.html')

@app.route('/pencegahan/snot')
def pencegahan_snot():
    return render_template('pencegahan_snot.html')

@app.route('/pencegahan/kolera')
def pencegahan_kolera():
    return render_template('pencegahan_kolera.html')

@app.route("/riwayat")
def riwayat():
    if 'loggedin' not in session or not session['loggedin']:
        flash('Silakan login terlebih dahulu untuk melihat riwayat prediksi', 'warning')
        return redirect(url_for('login'))
    
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM riwayat_prediksi WHERE username = %s ORDER BY tanggal DESC', (session['username'],))
        history = cursor.fetchall()
        cursor.close()

        username = session.get('username')
        
        return render_template("riwayat.html", history=history, username=username)
    except Exception as e:
        logging.error(f"Error fetching prediction history: {str(e)}")
        flash(f'Terjadi kesalahan saat mengambil data riwayat: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/get-chart-data')
def get_chart_data():
    if 'loggedin' not in session or not session['loggedin']:
        return jsonify({"error": "Belum login"}), 401

    username = session.get('username')

    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT hasil FROM riwayat_prediksi WHERE username = %s", (username,))
        hasil_rows = cursor.fetchall()
        cursor.close()

        # Hitung frekuensi tiap hasil penyakit
        hasil_list = [row['hasil'] for row in hasil_rows]
        count = Counter(hasil_list)

        labels = list(count.keys())
        values = list(count.values())

        return jsonify({
            "labels": labels,
            "values": values
        })
    except Exception as e:
        logging.error(f"Error fetching chart data: {str(e)}")
        return jsonify({"error": "Gagal mengambil data chart"}), 500


@app.context_processor
def utility_processor():
    
    return {
        'penyakit_mapping': penyakit_mapping,
        'deskripsi_penyakit': deskripsi_penyakit,
        'gambar_penyakit': gambar_penyakit
    }

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.", "success")
    return redirect(url_for('login'))


def render_nocache_template(template_name, **context):
    response = make_response(render_template(template_name, **context))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def setup_logging():
    if not os.path.exists('logs'):
        os.mkdir('logs')

    log_filename = f'logs/app-{datetime.now().strftime("%Y%m%d")}.log'
    error_log_filename = f'logs/error-{datetime.now().strftime("%Y%m%d")}.log'
    
    # Configure logging level and format
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.FileHandler(error_log_filename),
            logging.StreamHandler()  # Output to console
        ]
    )

    logging.info("Application started")

# Call setup_logging at the bottom before app.run
setup_logging()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)