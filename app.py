from flask import Flask, render_template, request, jsonify
import os
import sqlite3
import random
from werkzeug.utils import secure_filename
from classification.classify import human_or_not

import face_recognition
import cv2
import numpy as np
from flask import send_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

KNOWN_IMAGE_PATH = 'person.jpeg'
known_face_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(KNOWN_IMAGE_PATH))[0]]


phrases = [
    "The quick brown fox jumps over the lazy dog.",
    "Flask is a lightweight WSGI web application framework.",
    "Never trust a computer you can’t throw out a window.",
    "Security is not a product but a process.",
    "Speak friend and enter."
]

@app.route('/check_face', methods=['POST'])
def check_face():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame uploaded'}), 400

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            return jsonify({'match': True})

    return jsonify({'match': False})


def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        code TEXT,
        name BOOLEAN,
        phone_number BOOLEAN,
        dob BOOLEAN,
        ssn BOOLEAN,
        email_2fa BOOLEAN,
        captcha BOOLEAN,
        ishuman BOOLEAN,
        issamevoice BOOLEAN
    )''')
    conn.commit()
    conn.close()

@app.route('/get_phrase')
def get_phrase():
    return jsonify({'phrase': random.choice(phrases)})

@app.route('/rep')
def rep_page():
    code = str(random.randint(10000, 99999))
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (code, name, phone_number, dob, ssn, email_2fa, captcha, ishuman, issamevoice) VALUES (?, 0, 0, 0, 0, 0, 0, 0, 0)", (code,))
    conn.commit()
    conn.close()
    return render_template('rep.html', code=code)

@app.route('/customer')
def customer_page():
    return render_template('customer.html')

@app.route('/validate_code', methods=['POST'])
def validate_code():
    code = request.json['code']
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE code=?", (code,))
    row = c.fetchone()
    conn.close()
    return jsonify({'valid': bool(row)})

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    field = data['field']
    value = data['value']
    code = data['code']
    valid = False

    if field == 'name':
        first = value.get('first', '').strip().lower()
        middle = value.get('middle', '').strip().lower()
        last = value.get('last', '').strip().lower()
        if first == 'john' and middle == 'a' and last == 'doe':
            valid = True
    elif field == 'phone_number' and len(value) == 10 and value.isdigit():
        valid = True
    elif field == 'dob' and value.count('-') == 2:
        valid = True
    elif field == 'ssn' and len(value) == 9 and value.isdigit():
        valid = True
    elif field == 'email_2fa' and value == '0000':
        valid = True
    elif field == 'captcha' and value == '1234':
        valid = True

    if valid:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute(f"UPDATE users SET {field}=1 WHERE code=?", (code,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/set_decision', methods=['POST'])
def set_decision():
    data = request.json
    code = data['code']
    ishuman = data['ishuman']
    issamevoice = data['issamevoice']
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("UPDATE users SET ishuman=?, issamevoice=? WHERE code=?", (ishuman, issamevoice, code))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/status/<code>')
def status(code):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE code=?", (code,))
    row = c.fetchone()
    columns = [col[0] for col in c.description]
    conn.close()
    if row:
        return jsonify(dict(zip(columns, row)))
    else:
        return jsonify({'error': 'Code not found'})

@app.route('/check_human', methods=['POST'])
def check_human():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    is_human = human_or_not(file_path)
    return jsonify({'isHuman': is_human})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
