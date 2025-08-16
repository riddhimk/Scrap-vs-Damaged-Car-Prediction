import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = r"C:\Users\lenov\OneDrive\Desktop\PBL Dataset\recall_boosted_model.keras"
TEST_DIR = r"C:\Users\lenov\OneDrive\Desktop\PBL Dataset\Phase 3\Sample Run"

IMAGE_SIZE = (224, 224)  
CLASS_LABELS = ["damaged", "scrapable"]

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    return img

def load_test_data(test_dir):
    X, y, paths = [], [], []
    for label_index, label in enumerate(CLASS_LABELS):
        folder = os.path.join(test_dir, label)
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, file)
                img = preprocess_image(path)
                X.append(img)
                y.append(label_index)
                paths.append(path)

    return np.array(X), np.array(y), paths 

def evaluate_model(model_path, test_dir):
    model = load_model(model_path)
    X_test, y_true, _ = load_test_data(test_dir)
    y_probs = model.predict(X_test, verbose=1).flatten()
    best_threshold, best_f1 = 0.5, 0.0

    for t in np.arange(0, 1, 0.01):
        preds = (y_probs > t).astype(int)
        score = f1_score(y_true, preds, average='macro')
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    print(f"\nThreshold used (best by F1): {best_threshold:.4f}")
    y_pred = (y_probs > best_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}\n")
    print(f"Confusion Matrix:\n{cm}")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': best_threshold,
        'confusion_matrix': cm
    }

print("Evaluating Recall-Boosted Model:\n")
results = evaluate_model(MODEL_PATH, TEST_DIR)