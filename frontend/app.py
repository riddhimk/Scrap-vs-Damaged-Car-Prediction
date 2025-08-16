from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from livereload import Server

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(r"C:\Users\lenov\OneDrive\Desktop\TEMP\flask frontend\recall_boosted_model.keras")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]

    threshold = 0.41
    prob_scrapped = float(prediction)
    
    if prob_scrapped > threshold:
        result = "Scrapable"
        confidence_score = prob_scrapped * 100
    else:
        result = "Damaged"
        confidence_score = (1 - prob_scrapped) * 100

    return redirect(url_for('result', result=result,
                            confidence=f"Detection Score: {confidence_score:.2f}%",
                            image=filename))

@app.route('/result', methods=['GET', 'POST'])
def result():
    result = request.args.get('result')
    confidence = request.args.get('confidence')
    image = request.args.get('image')
    feedback_message = None

    if request.method == 'POST':
        feedback = request.form.get('message')
        
        if feedback:
            feedback_message = "Thank you for your feedback!"
        else:
            feedback_message = "Feedback is empty!"

    return render_template('result.html', result=result, confidence=confidence, image=image, feedback_message=feedback_message)

if __name__ == '__main__':
    server = Server(app.wsgi_app)
    server.watch('templates/*.html')   
    server.watch('static/*.*')         
    server.watch('*.py')               
    server.serve(port=5000, debug=True)
