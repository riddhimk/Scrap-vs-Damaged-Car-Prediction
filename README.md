# Scrap vs Damaged Car Prediction (GreenFleet)

A web application powered by deep learning that classifies a vehicle's condition as either **scrap** (end of life) or **damaged** (repairable). Built using **TensorFlow**, a tuned **EfficientNet** model, and a **Flask** interface for user-friendly interactions.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Usage](#usage)
* [Model Training Pipeline](#model-training-pipeline)
* [Model Deployment (Flask App)](#model-deployment-flask-app)
* [Requirements](#requirements)
* [Future Enhancements](#future-enhancements)
* [Authors](#authors)

---

## Project Structure

```
GreenFleet/
│
├── frontend/
│   ├── static/               
│   ├── templates/            
│   ├── uploads/              
│   ├── app.py                # Flask API for web UI and prediction
│   ├── test.py               # Quick local tests for model inference
│   └── recall_boosted_model.keras  # Pre-trained model (tracked using Git LFS)
│
├── model_training.py         # Code for training, tuning, and fine-tuning the model
├── requirements.txt          # Exact Python packages & versions
└── README.md                 
```

---

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/riddhimk/Scrap-vs-Damaged-Car-Prediction.git
   cd Scrap-vs-Damaged-Car-Prediction
   ```

2. **Install dependencies (ensure Git LFS is installed)**

   ```bash
   git lfs install
   pip install -r requirements.txt
   ```

3. **Run the Flask app**

   ```bash
   cd frontend
   python app.py
   ```

   Navigate to `http://127.0.0.1:5000` in your browser, upload a car image, and see the prediction result.

---

## Usage

* **Upload Interface**: The Flask app allows users to upload images via a form and returns a prediction—either "Scrap" or "Damaged".
* **test.py**: Use this script to run local tests on images using the model, without launching the UI.

---

## Model Training Pipeline

* Data is augmented using `ImageDataGenerator` (rotations, flips, zooms, brightness variations).
* **EfficientNetB1** is used as a base (pre-trained on ImageNet) with custom dense layers.
* Hyperparameters (units, dropout rates, learning rates) are tuned using **Keras Tuner (Hyperband)**.
* Once best parameters are found, the model is fine-tuned end-to-end.
* Scripts: `model_training.py` → trains models and outputs a `.keras` model file; hyperparameter tuning and evaluation are embedded within.

---

## Requirements

Install dependencies via pip:

```
flask==3.1.1
keras==3.8.0
kt_legacy==1.0.5
livereload==2.7.1
matplotlib==3.8.0
numpy==2.3.2
opencv_python==4.10.0.84
opencv_python_headless==4.10.0.84
pandas==2.3.1
scikit_learn==1.4.0
seaborn==0.13.2
tensorflow==2.18.0
Werkzeug==3.1.3
```

To install:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements

* **Deployment**: Use Docker or Streamlit to package and deploy the app easily.
* **Extended Classes**: Expand classification from binary to multi-class (e.g., “Minor Damage”, “Moderate Damage”, “Scrap”).
* **Continuous Training**: Add CI/CD pipeline to retrain model automatically when dataset updates.
* **Evaluation Metrics**: Add confusion matrix, ROC curves, and AUC logging to better monitor performance.

---

## Authors

* **Riddhim Kawdia** — [@riddhimk](https://github.com/riddhimk)
* **Kartik Mehta** — [@MeWan08](https://github.com/MeWan08)
* **Karan Rekhan** — [@KaranRekhan](https://github.com/KaranRekhan)
* **Veeraj Kumar Sahu** — [@Veeraj77](https://github.com/Veeraj77)
