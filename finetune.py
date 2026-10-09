import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

def finetune_and_evaluate():
    print("Loading data...")
    # Load labels
    df = pd.read_csv('data/labels.csv')
    df['label'] = df['label'].astype(str)
    
    # Preprocessing generator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    model_path = os.path.join('model', 'recall_boosted_model.keras')
    print(f"Loading original model from {model_path}...")
    model = load_model(model_path)
    
    input_shape = model.input_shape
    if len(input_shape) == 4:
        target_size = (input_shape[1], input_shape[2])
    else:
        target_size = (240, 240)
    
    print(f"Model target size: {target_size}")
    
    batch_size = 32
    
    # Train generator
    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory='data/images/',
        x_col='filename',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    # Validation generator
    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory='data/images/',
        x_col='filename',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    print("Evaluating BEFORE fine-tuning...")
    val_gen.reset()
    preds_before = model.predict(val_gen, steps=len(val_gen))
    preds_before_class = (preds_before.flatten() > 0.41).astype(int)
    y_true = val_gen.classes
    
    acc_before = accuracy_score(y_true, preds_before_class)
    f1_before = f1_score(y_true, preds_before_class)
    print(f"Metrics Before FT: Accuracy: {acc_before:.4f}, F1: {f1_before:.4f}")
    
    print("\nComputing Class Weights...")
    train_classes = train_gen.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_classes),
        y=train_classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {class_weight_dict}")
    
    print("\nFine-tuning the model with Class Balancing...")
    # Compile with low learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
                  
    # Train for 1 epoch
    model.fit(
        train_gen,
        epochs=1,
        validation_data=val_gen,
        class_weight=class_weight_dict
    )
    
    print("\nEvaluating AFTER fine-tuning with class weights...")
    val_gen.reset()
    preds_after = model.predict(val_gen, steps=len(val_gen))
    preds_after_class = (preds_after.flatten() > 0.41).astype(int)
    
    acc_after = accuracy_score(y_true, preds_after_class)
    f1_after = f1_score(y_true, preds_after_class)
    
    print("\n" + "="*50)
    print("CLASS-BALANCED FT PERFORMANCE METRICS (Threshold=0.41)")
    print("="*50)
    print(f"Accuracy Before: {acc_before:.4f}  ->  After: {acc_after:.4f}")
    print(f"F1 Score Before: {f1_before:.4f}  ->  After: {f1_after:.4f}")
    print("\nClassification Report (After FT):")
    print(classification_report(y_true, preds_after_class, target_names=['Damaged', 'Scrapable']))
    print("="*50)
    
    # Not saving model per instructions

if __name__ == '__main__':
    finetune_and_evaluate()

