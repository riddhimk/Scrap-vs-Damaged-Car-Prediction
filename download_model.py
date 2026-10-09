import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

def download_lightweight_model():
    print("Downloading lightweight MobileNetV2 base model...")
    # MobileNetV2 is one of the most efficient, lightweight models available
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom head for our binary classification (Damaged vs Scrapable)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Save it over the old one or as a new one
    model_path = os.path.join('model', 'recall_boosted_model.keras')
    
    # Check if we should backup the old model
    if os.path.exists(model_path):
        backup_path = model_path.replace('.keras', '_old.keras')
        os.rename(model_path, backup_path)
        print(f"Backed up old model to {backup_path}")
        
    model.save(model_path)
    print(f"Successfully saved new lightweight model to {model_path}")

if __name__ == "__main__":
    download_lightweight_model()
