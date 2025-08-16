import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Define the path to your dataset directory
DATASET_DIR = r"/Users/karanrekhan/Downloads/PBL Dataset/New Labeled Dataset"

# Function to create data generators
def get_data_generators(batch_size, data_dir):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(240, 240),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(240, 240),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen

# Model-building function
def build_model(hp):
    base_model = keras.applications.EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=(240, 240, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(240, 240, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    for i in range(hp.Int('num_layers', 1, 4)):
        x = layers.Dense(
            units=hp.Int(f'units_{i}', min_value=128, max_value=1024, step=128),
            activation='relu'
        )(x)
        x = layers.Dropout(
            hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.6, step=0.1)
        )(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    return model

# Plot tuner results
def plot_hyperparameter_results(tuner):
    trials = tuner.oracle.get_best_trials(num_trials=50)
    data = []

    for trial in trials:
        hp_dict = trial.hyperparameters.values
        hp_dict['score'] = trial.score
        data.append(hp_dict)

    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(df.columns):
        if col != 'score' and df[col].dtype in [np.int64, np.float64]:
            plt.subplot(2, 3, i+1)
            plt.scatter(df[col], df['score'], alpha=0.6)
            plt.xlabel(col)
            plt.ylabel('Validation Accuracy')
            plt.title(f'{col} vs Val Accuracy')
    plt.tight_layout()
    plt.suptitle("Hyperparameter Tuning Results", fontsize=16, y=1.02)
    plt.show()

# Plot ROC Curve
def plot_roc_curve(model, val_gen):
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Main training loop
def train_model():
    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='tuner_dir',
        project_name='car_damage',
        overwrite=True
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        min_delta=0.001
    )

    best_accuracy = 0
    best_model = None
    best_tuner = None
    best_batch_size = None

    for batch_size in [8, 16, 24]:
        print(f"\nTesting batch size: {batch_size}")
        train_gen, val_gen = get_data_generators(batch_size, DATASET_DIR)

        tuner.search(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            callbacks=[early_stop],
            verbose=1
        )

        model = tuner.get_best_models(num_models=1)[0]
        val_loss, val_acc, val_precision, val_recall = model.evaluate(val_gen, verbose=0)
        print(f"Batch {batch_size} | Val Acc: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_tuner = tuner
            best_batch_size = batch_size

    # Save the best model before fine-tuning
    best_model.save('best_car_damage_model.h5')

    # Fine-tuning
    best_model.trainable = True
    for layer in best_model.layers[1].layers[-20:]:
        layer.trainable = True

    best_model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    lr_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )

    train_gen, val_gen = get_data_generators(best_batch_size, DATASET_DIR)
    best_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=[early_stop, lr_reduction]
    )

    best_model.save('fine_tuned_car_damage_model.h5')
    print("\nTraining complete. Models saved.")

    # Plotting results
    if best_tuner:
        plot_hyperparameter_results(best_tuner)

    plot_roc_curve(best_model, val_gen)

if __name__ == "__main__":
    train_model()