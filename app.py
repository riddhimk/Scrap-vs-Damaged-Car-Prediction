import os
import json
import uuid
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename

from services.predictor import VehicleDamagePredictor
from services.preprocessing import validate_image_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOVABLE_PUBLIC = os.path.join(BASE_DIR, 'lovable-ui', '.output', 'public')

app = Flask(
    __name__,
    static_folder='static',
    static_url_path=''
)
app.secret_key = 'greenfleet_super_secret_key'

if os.environ.get('VERCEL') or not os.access('.', os.W_OK):
    app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'greenfleet_uploads')
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

try:
    predictor = VehicleDamagePredictor()
except Exception as e:
    print('Failed to initialize Gemini predictor:', e)
    predictor = None

try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
except Exception as e:
    print('Upload folder creation note:', e)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/')
def index():
    p = os.path.join(LOVABLE_PUBLIC, 'index.html')
    if os.path.exists(p):
        return send_from_directory(LOVABLE_PUBLIC, 'index.html')
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    p = os.path.join(LOVABLE_PUBLIC, 'index.html')
    if os.path.exists(p):
        return send_from_directory(LOVABLE_PUBLIC, 'index.html')
    return render_template('index.html')

@app.route('/analyzing')
def analyzing_page():
    p = os.path.join(LOVABLE_PUBLIC, 'index.html')
    if os.path.exists(p):
        return send_from_directory(LOVABLE_PUBLIC, 'index.html')
    return render_template('index.html')

@app.route('/results')
def results_page():
    p = os.path.join(LOVABLE_PUBLIC, 'index.html')
    if os.path.exists(p):
        return send_from_directory(LOVABLE_PUBLIC, 'index.html')
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    p = os.path.join(LOVABLE_PUBLIC, 'favicon.ico')
    folder = LOVABLE_PUBLIC if os.path.exists(p) else 'static'
    resp = send_from_directory(folder, 'favicon.ico', mimetype='image/x-icon', max_age=0)
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    p = os.path.join(LOVABLE_PUBLIC, 'assets')
    if os.path.exists(os.path.join(p, filename)):
        return send_from_directory(p, filename)
    return send_from_directory(os.path.join('static', 'assets'), filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def api_analyze():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if predictor is None:
        return jsonify({'error': 'AI Engine not initialized.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    f = request.files['file']
    is_valid, err = validate_image_file(f)
    if not is_valid:
        return jsonify({'error': err}), 400

    fn = secure_filename(f.filename)
    unique_name = str(uuid.uuid4().hex) + '_' + fn
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    f.save(path)

    try:
        result = predictor.predict(path)
        pred = result['prediction']
        if 'Undamaged' in pred:
            cat = 'Undamaged'
        elif 'Scrapable' in pred:
            cat = 'Scrapable'
        else:
            cat = 'Damaged'

        conf = float(result.get('confidence_score', 95.0))
        rem = max(0.0, 100.0 - conf) / 2.0
        scores = {c: f'{rem:.1f}%' for c in ['Damaged', 'Undamaged', 'Scrapable']}
        scores[cat] = f'{conf:.1f}%'

        return jsonify({
            'classification': cat,
            'confidenceScore': f'{conf:.1f}%',
            'confidencePercent': conf,
            'breakdown': [[c, scores[c]] for c in ['Damaged', 'Undamaged', 'Scrapable']],
            'summary': (str(result.get('detailed_reasoning', '')) + ' Verdict: ' + str(result.get('damage_severity', ''))).strip(),
            'filename': unique_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
