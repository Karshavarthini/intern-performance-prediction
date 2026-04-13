import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "trained_model.pkl"))
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "scaler.pkl"))

# --- Load Assets ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Intelligence System: Online")
except Exception as e:
    print(f"❌ Load Error: {e}")
    model, scaler = None, None

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

results_map = {2: "High", 1: "Medium", 0: "Low"}

# ──────────────────────────────────────────────
# SHARED HELPER: build the 13-feature array
# Training column order:
#   0  attendance_percentage
#   1  consistency_score
#   2  learning_score
#   3  tasks_assigned        → always 20 in dataset
#   4  tasks_completed
#   5  completion_rate       → tasks_completed / active_days
#   6  project_score
#   7  evaluation_score
#   8  login_frequency
#   9  active_days
#   10 engagement_score
#   11 communication_score
#   12 discipline_score
# ──────────────────────────────────────────────
def build_features_13(
    attendance, consistency, learning,
    tasks_assigned, tasks_completed,
    project_score, evaluation,
    login_frequency, active_days,
    engagement, communication, discipline
) -> np.ndarray:
    """Return a (1, 13) numpy array ready for scaler.transform()."""
    active_days_safe = max(active_days, 1)
    completion_rate = tasks_completed / active_days_safe

    features_list = [
        attendance,       # 0
        consistency,      # 1
        learning,         # 2
        tasks_assigned,   # 3
        tasks_completed,  # 4
        completion_rate,  # 5
        project_score,    # 6
        evaluation,       # 7
        login_frequency,  # 8
        active_days,      # 9
        engagement,       # 10
        communication,    # 11
        discipline        # 12
    ]
    return np.array([features_list])


# ──────────────────────────────────────────────
# HOME  ← THIS WAS THE MISSING ROUTE
# ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": None,
            "batch_results": None,
            "error": None
        }
    )


# ──────────────────────────────────────────────
# SINGLE PREDICTION
# ──────────────────────────────────────────────
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    attendance:      float = Form(...),
    consistency:     float = Form(...),
    learning:        float = Form(...),
    tasks_assigned:  float = Form(20),
    tasks_completed: float = Form(...),
    project_score:   float = Form(...),
    evaluation:      float = Form(...),
    login_frequency: float = Form(...),
    active_days:     float = Form(...),
    engagement:      float = Form(...),
    communication:   float = Form(...),
    discipline:      float = Form(...)
):
    result = "Error"
    if model and scaler:
        try:
            features = build_features_13(
                attendance, consistency, learning,
                tasks_assigned, tasks_completed,
                project_score, evaluation,
                login_frequency, active_days,
                engagement, communication, discipline
            )
            scaled   = scaler.transform(features)
            raw_pred = int(model.predict(scaled)[0])
            result   = results_map.get(raw_pred, "Unknown")

            print(f"DEBUG Single: Raw={raw_pred} | Result={result}")

        except Exception as e:
            print(f"❌ Single Predict Error: {e}")
            result = f"Error: {e}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result,
            "batch_results": None,
            "error": None
        }
    )


# ──────────────────────────────────────────────
# BATCH PREDICTION
# CSV format (no header, 12 values per row):
#   attendance, consistency, learning,
#   tasks_assigned, tasks_completed,
#   project_score, evaluation,
#   login_frequency, active_days,
#   engagement, communication, discipline
# ──────────────────────────────────────────────
BATCH_COLS = [
    "attendance_percentage", "consistency_score", "learning_score",
    "tasks_assigned", "tasks_completed", "project_score",
    "evaluation_score", "login_frequency", "active_days",
    "engagement_score", "communication_score", "discipline_score"
]

FEATURE_ORDER = [
    "attendance_percentage", "consistency_score", "learning_score",
    "tasks_assigned", "tasks_completed", "completion_rate",
    "project_score", "evaluation_score", "login_frequency", "active_days",
    "engagement_score", "communication_score", "discipline_score"
]

@app.post("/predict_batch", response_class=HTMLResponse)
async def predict_batch(request: Request, batch_input: str = Form(...)):
    if not model or not scaler:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error": "Model offline",
                "result": None,
                "batch_results": None
            }
        )

    try:
        # 1. Parse lines
        lines = [l.strip() for l in batch_input.strip().splitlines() if l.strip()]

        # Skip optional header row
        if lines and not lines[0].replace(",", "").replace(".", "").replace("-", "").strip().replace(" ", "").isdigit():
            lines = lines[1:]

        if not lines:
            raise ValueError("No data rows found. Paste numeric CSV rows (no header needed).")

        rows = [row.split(",") for row in lines]

        # 2. Build raw DataFrame (12 input columns)
        df = pd.DataFrame(rows, columns=BATCH_COLS).astype(float)

        # 3. Compute completion_rate
        df["active_days_safe"] = df["active_days"].clip(lower=1)
        df["completion_rate"]  = df["tasks_completed"] / df["active_days_safe"]

        # 4. Assemble in exact training column order (13 cols)
        X         = df[FEATURE_ORDER].values   # shape (n, 13)
        X_scaled  = scaler.transform(X)
        raw_preds = model.predict(X_scaled)

        df["Prediction"] = [results_map.get(int(p), "Unknown") for p in raw_preds]

        print(f"DEBUG Batch: {len(df)} rows | {df['Prediction'].value_counts().to_dict()}")

        # 5. Render results table
        table_html = df[[
            "attendance_percentage", "tasks_completed",
            "evaluation_score", "Prediction"
        ]].to_html(classes="batch-table", index=True, border=0)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "batch_results": table_html,
                "result": None,
                "error": None
            }
        )

    except ValueError as ve:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "batch_results": None,
                "result": None,
                "error": str(ve)
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "batch_results": None,
                "result": None,
                "error": f"Batch Error: {e}"
            }
        )