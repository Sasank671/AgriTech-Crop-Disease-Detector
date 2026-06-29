# 🌿 KrishiAI — AgriTech Crop Disease Detector

Upload a photo of a crop leaf → get the disease, a severity estimate, and a
**TNAU-sourced** pesticide treatment recommendation, in **English or Hindi**.

Built for farmer-facing use: bilingual UI, no English-literacy requirement, and
**deliberately no LLM** in the advice path — treatment text comes from a curated,
human-reviewable lookup table to eliminate hallucination risk in safety-critical
agricultural guidance.

## Highlights
- **EfficientNet-B3 (timm)** trained on the full **38-class PlantVillage** dataset
  — **99.8% test accuracy**, macro-F1 0.997.
- Exported to **ONNX** for fast CPU inference (served via ONNX Runtime).
- **FastAPI** backend with **JWT auth** and a **PostgreSQL** user + scan-history database.
- Bilingual (**English / हिंदी**) vanilla-JS frontend, including Hindi treatment
  instructions and severity (pesticide names & dosages stay in English by design).
- Per-user **scan history** — click any past scan to re-open its treatment.

## Tech stack
PyTorch · EfficientNet-B3 · FastAPI · ONNX · PostgreSQL · SQLAlchemy · JWT · JavaScript

## Project structure
```
.
├── main.py                # FastAPI backend (auth, /predict, /history, ONNX inference)
├── train.py               # EfficientNet-B3 trainer (Kaggle-ready) + ONNX export
├── treatments.json        # TNAU-sourced treatment lookup (EN + curated Hindi)
├── requirements.txt       # serving deps
├── requirements-train.txt # training deps
├── .env.example           # copy to .env (DATABASE_URL, SECRET_KEY)
├── krishiai/              # frontend
│   ├── index.html
│   ├── app.js
│   └── style.css
└── output/                # model artifacts
    ├── model.onnx         # served model (+ model.onnx.data external weights)
    ├── class_names.json   # index → class name (must match the model)
    ├── metrics.json
    ├── confusion_matrix.png
    └── training_curves.png
```

## Setup
```bash
# 1. Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env          # then edit DATABASE_URL + SECRET_KEY

# 3. Create the database (Postgres must be running)
createdb krishiai             # tables are auto-created on first startup

# 4. Run
uvicorn main:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000/** → register → pick language → upload a leaf image.

## Retraining (38 classes)
`train.py` is written to run on a **Kaggle GPU notebook**:
1. Add the `plantvillage-dataset` dataset, enable GPU + Internet.
2. `pip install timm onnx onnxscript onnxruntime`, then run `train.py`.
3. Copy `model.onnx`, `model.onnx.data`, `class_names.json`, `metrics.json` from
   `/kaggle/working/output/` into this repo's `output/`.

## Model performance
| Metric | Value |
|---|---|
| Test accuracy | 99.8% |
| Macro F1 | 0.997 |
| Classes | 38 |
| Test images | 5,431 |

## Data & credits
- Dataset: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (38-class color set).
- Treatments: [TNAU Agritech Portal](https://agritech.tnau.ac.in/).
