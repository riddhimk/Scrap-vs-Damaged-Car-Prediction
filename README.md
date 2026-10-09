# GreenFleet: AI-Based Vehicle Damage Assessment

## Project Overview
GreenFleet is a web application that performs preliminary assessment of vehicle damage using deep learning. It allows users to upload a photo of a vehicle and instantly classify its condition as either **Damaged** (repairable) or **Scrapable** (end-of-life).

## Business Problem & Solution
Insurance adjusters and salvage yards often spend significant time categorizing vehicles after accidents. GreenFleet provides a rapid, AI-driven first pass to automatically flag whether a vehicle is likely salvageable or headed for the scrap yard. This improves triaging efficiency and streamlines workflows.

## Model Description
The core inference engine uses an EfficientNet-based Convolutional Neural Network (CNN). 
**Note:** This project reuses an existing, pre-trained model (`recall_boosted_model.keras`). **The model was not retrained during this project.** We simply loaded the existing static weights and improved the application architecture, preprocessing consistency, and user experience around it.

## Installation and Setup

1. **Clone the repository and set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Web UI**
   Open your browser and navigate to the local URL: `http://127.0.0.1:5000`

## Project Structure
```text
GreenFleet/
├── app.py                  # Main Flask application and routes
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── model/                  # Directory containing the pre-trained Keras model
├── services/               # Core logic decoupled from routing
│   ├── predictor.py        # Model loading, threshold evaluation, and prediction
│   ├── preprocessing.py    # Robust image validation and dynamic resizing
│   └── tta.py              # Test-Time Augmentation logic
├── templates/              # HTML templates (index, result)
├── static/                 # CSS styling and Client-side JS
├── uploads/                # Temporary directory for user uploads
├── evaluate_model.py       # Standalone evaluation script
└── feedback.json           # User feedback store
```

## Evaluation Instructions
To evaluate the static model against new labeled validation data:
1. Organize your test images into two folders: `data/val/damaged/` and `data/val/scrap/`.
2. Run the evaluation script:
   ```bash
   python evaluate_model.py
   ```
3. The script will dynamically read the expected model input size, process the images, and output key metrics such as Accuracy, F1-Score, Confusion Matrix, and ROC-AUC. It also compares the default threshold (`0.41`) against standard thresholds (`0.50`).

## Limitations
* **Preliminary Assessment Only:** This tool is not a substitute for a professional, in-person vehicle inspection.
* **Sensitivity to Angles/Quality:** The model's accuracy heavily depends on lighting, image resolution, and whether the damaged section of the vehicle is clearly visible in the photo.
* **Domain Shift:** Predictions rely on the distribution of the original training data. Vehicles that look drastically different from the training set may yield unpredictable results.
