from flask import Flask, render_template, redirect, url_for, jsonify, request
import pandas as pd
from main import run_attendance
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run')
def run():
    result = run_attendance()
    return render_template('run_result.html', result=result)

@app.route('/attendance')
def attendance():
    try:
        df = pd.read_excel('attendance.xlsx')
        records = df.to_dict(orient='records')
    except Exception as e:
        records = []
    return render_template('attendance.html', records=records)

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({'status': 'error', 'message': 'No name provided'}), 400
    try:
        import os
        import pandas as pd
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        if os.path.exists('attendance.xlsx'):
            df = pd.read_excel('attendance.xlsx')
        else:
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df = pd.concat([df, pd.DataFrame({'Name': [name], 'Date': [date_str], 'Time': [time_str]})], ignore_index=True)
        df.to_excel('attendance.xlsx', index=False)
        return jsonify({'status': 'success', 'name': name, 'date': date_str, 'time': time_str})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/face_labels')
def face_labels():
    faces_dir = os.path.join(app.root_path, 'static', 'faces')
    if not os.path.exists(faces_dir):
        return jsonify([])
    labels = [name for name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, name))]
    return jsonify(labels)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 