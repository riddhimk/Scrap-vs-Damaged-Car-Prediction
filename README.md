# GreenFleet: AI-Based Vehicle Damage Assessment

[![Live Demo](https://img.shields.io/badge/Live%20Demo-greenfleet--rust.vercel.app-brightgreen?logo=vercel)](https://greenfleet-rust.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)

> 🌐 **Live Cloud Application**: [https://greenfleet-rust.vercel.app/](https://greenfleet-rust.vercel.app/)  
> 📁 **Legacy Code & Artifacts**: Historical training scripts and prior model files are preserved in [`old/`](./old).

---

## Project Overview
**GreenFleet** is an AI-powered vehicle assessment platform designed for instant automotive damage screening. Drivers, fleet operators, and insurance adjusters can upload a single vehicle photo and immediately receive:
- **Condition Classification**: Categorized into **Undamaged (Pristine)**, **Damaged (Repairable)**, or **Scrapable (Unrepairable / Total Loss)**.
- **Confidence Score & Breakdown**: Granular probability distribution across vehicle conditions.
- **Detailed Damage Reasoning**: Visual evidence summary highlighting damaged components, impact severity, and repairability.

---

## Architecture & Technology Stack
- **Frontend**: Modern SPA built with high-performance responsive UI components, real-time assessment states, and custom branding.
- **Backend**: Python Flask application running serverlessly via `@vercel/python` on Vercel.
- **AI Engine**: Multimodal Vision Language Model powered by Google Gemini API (`gemini-flash-lite-latest` / `gemini-2.5-flash-lite`), paired with a robust fallback vision engine.
- **Zero-Cost Academic Cloud Deployment**: Hosted completely free on Vercel with zero cold-storage lock-in and secured environment variables.

---

## Project Structure
```text
GreenFleet/
├── app.py                  # Main Flask application and serverless routes
├── requirements.txt        # Lightweight, serverless-optimized dependencies
├── vercel.json             # Vercel serverless runtime & routing configuration
├── .vercelignore           # Excludes heavy legacy archives from cloud builds
├── README.md               # Project documentation and deployment details
├── services/               # Decoupled AI & vision services
│   ├── predictor.py        # Gemini multimodal reasoning & fallback predictor
│   ├── preprocessing.py    # Image validation, security sanitization, resizing
│   └── tta.py              # Test-Time Augmentation utilities
├── static/                 # Production precompiled SPA assets, CSS, icons
├── templates/              # Production HTML templates
├── lovable-ui/             # Source UI components and design system
└── old/                    # Preserved historical repository files & scripts
    ├── frontend/           # Original legacy UI
    ├── mode_training.py    # Original CNN training script
    └── Demo Video.mp4      # Original project demonstration video
```

---

## Local Installation and Setup

### 1. Clone the repository and create a virtual environment
```bash
git clone https://github.com/riddhimk/Scrap-vs-Damaged-Car-Prediction.git
cd Scrap-vs-Damaged-Car-Prediction
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key
Create an `api.txt` file in the project root (or set the `GEMINI_API_KEY` environment variable):
```text
<YOUR_GEMINI_API_KEY>
```

### 4. Run the Application locally
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## Cloud Deployment (Vercel)
This repository is pre-configured with `vercel.json` for one-click deployment:
1. Import repository into **Vercel**.
2. Select **Framework Preset**: `Other`.
3. Set **Environment Variable**: `GEMINI_API_KEY = <your-api-key>`.
4. Click **Deploy**.

---

## Disclaimer
GreenFleet provides preliminary, assistive damage triage from visual imagery and does not replace certified in-person structural or mechanical automotive inspections.
