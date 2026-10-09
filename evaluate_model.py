import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress OneDNN warnings
import warnings
warnings.filterwarnings('ignore')         # Suppress Python warnings

import logging
logging.getLogger('absl').setLevel(logging.ERROR)


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from services.predictor import VehicleDamagePredictor

def evaluate(data_dir='data/val', batch_size=32):
    """
    Evaluates the model on a directory of labeled validation/test data.
    Expected structure:
    data_dir/
       ├── damaged/
       └── scrap/
    """
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        print("Please structure your labeled data as:")
        print(f"  {data_dir}/")
        print(f"    ├── damaged/  (class 0)")
        print(f"    └── scrap/    (class 1)")
        return

    # Load predictor to dynamically get shape and model
    predictor = VehicleDamagePredictor()
    model = predictor.model
    target_size = predictor.target_size
    
    print(f"\nEvaluating model with input shape: {target_size}")
    
    # We use ImageDataGenerator for easy loading, scaling by 1/255.0
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Load images without shuffling to keep labels aligned with predictions
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    if generator.samples == 0:
        print("No images found for evaluation.")
        return
        
    print(f"Found {generator.samples} images belonging to {generator.num_classes} classes.")
    print(f"Class mapping: {generator.class_indices}")
    
    # Predict probabilities
    print("\nRunning inference...")
    probs = model.predict(generator, steps=np.ceil(generator.samples/batch_size))
    probs = probs.flatten()
    
    y_true = generator.classes
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # Function to calculate and print metrics for a specific threshold
    def evaluate_threshold(thresh):
        print(f"\n--- Metrics @ Threshold: {thresh} ---")
        y_pred = (probs > thresh).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Damaged', 'Scrapable']))
        
    # Evaluate at default threshold
    evaluate_threshold(0.41)
    
    # Evaluate at standard 0.50 threshold for comparison
    evaluate_threshold(0.50)
    
    # Calculate ROC-AUC (independent of threshold)
    roc_auc = roc_auc_score(y_true, probs)
    print("\n--- Threshold-Independent Metrics ---")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("="*50)

if __name__ == '__main__':
    evaluate()
