# GreenFleet Recreation Prompt for Claude

Recreate my existing **GreenFleet: AI-Based Vehicle Damage Assessment** project as a polished, working web application using the existing trained model.

Reference repository:
https://github.com/riddhimk/Scrap-vs-Damaged-Car-Prediction

## Critical constraint

**Do NOT retrain the model.** Reuse the existing trained model for inference. Do not create a new training pipeline.

The existing model is:

`frontend/recall_boosted_model.keras`

Preserve the original project's purpose and prediction behaviour, while improving the application architecture, preprocessing consistency, UX, robustness, and inference-time handling.

## Project objective

Users upload a vehicle image and the system classifies it as:

- **Damaged**: damage appears repairable
- **Scrapable**: damage appears severe enough for end-of-life/scrap classification

This is a preliminary AI assessment, not a professional vehicle inspection.

## Existing ML system

The project uses a CNN/EfficientNet-based binary image classifier and Flask.

Load the existing `.keras` model directly.

After loading it, inspect `model.input_shape` and dynamically resize uploaded images to the model's actual expected dimensions. Do not hardcode 224x224 or 240x240 because the original repository has an input-size inconsistency.

Preserve the preprocessing expected by the trained model. Do not arbitrarily change normalization.

The original application uses a sigmoid-style binary prediction with a default threshold around **0.41**. Keep `0.41` as the default, but make the threshold configurable.

## Improve robustness/accuracy without retraining

Explore only inference-time improvements:

1. **Correct preprocessing**
   - Match the model's actual input shape.
   - Keep preprocessing deterministic and consistent.

2. **Optional Test-Time Augmentation (TTA)**
   - Original image
   - Horizontal flip
   - Small brightness/contrast variations
   - Average the predictions
   - Make TTA optional because it increases inference time.
   - Do not use aggressive transformations.

3. **Threshold evaluation**
   - If labelled validation/test data is available, evaluate thresholds around 0.41.
   - Compare accuracy, precision, recall and F1.
   - Change the default threshold only when actual evaluation supports it.
   - Never claim an improvement without measuring it.

4. **Confidence handling**
   - Detect predictions close to the decision boundary.
   - Show a manual-review/low-confidence warning for borderline cases.
   - Do not present uncertain predictions as definitive.

5. **Optional Grad-CAM**
   - If compatible with the existing model, add an optional Grad-CAM visualization.
   - Clearly label it as an activation visualization, not damage localization.

## Recommended structure

```text
GreenFleet/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   └── recall_boosted_model.keras
├── services/
│   ├── predictor.py
│   ├── preprocessing.py
│   └── tta.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── uploads/
└── evaluate_model.py
```

Keep inference logic separate from Flask routes. Use project-relative paths so the app works on different machines.

## Web interface

Create a modern professional UI combining automotive, sustainability and AI themes.

Landing page:
- GreenFleet branding
- Short product explanation
- Drag-and-drop/image upload
- Image preview
- Analyze button
- Supported file formats
- Loading state

Result page:
- Uploaded image
- Prediction: **DAMAGED** or **SCRAPABLE**
- Probability/score
- Confidence category
- Manual-review warning when appropriate
- Analyze Another Image button

Do not invent explanations such as specific damaged parts because this is a binary classifier, not a damage-localization model.

## Input validation

Handle:
- PNG/JPG/JPEG validation
- Corrupt/unreadable images
- Very small images
- Maximum upload size
- Safe filenames
- Unique upload names

Show clear errors instead of crashing.

## Feedback

Allow users to submit feedback after a prediction. Store it locally using JSON or SQLite.

Capture:
- prediction
- score/confidence
- timestamp
- feedback
- optional comment

Do not add unnecessary cloud services.

## Evaluation

Create `evaluate_model.py` that **only evaluates the existing model**.

When labelled test/validation data is available, calculate:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Also support threshold comparison.

Never fabricate metrics.

## README

Include:
- Project overview
- Business problem
- Solution
- Model description
- Explicit statement that the existing model is reused and not retrained
- Installation and virtual-environment setup
- `pip install -r requirements.txt`
- `python app.py`
- Local URL
- Project structure
- Evaluation instructions
- Limitations

State that predictions are preliminary and depend on image quality and training-data distribution.

## Final requirement

Do not stop at pseudocode. Produce a complete runnable implementation.

First inspect the reference repository and existing model/application structure, then recreate the application around the existing `.keras` model.

**Do not retrain the model.**
