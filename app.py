import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, Response, jsonify, render_template, request
from src.features import CHOICES, FEATURES, NUMERIC_INPUTS, add_features

BASE = Path(__file__).resolve().parent

# Load everything once at startup
model = joblib.load(BASE / "models/churn_model.pkl")
THRESHOLD = json.loads((BASE / "models/threshold.json").read_text(encoding="utf-8"))["threshold"]
METRICS = json.loads((BASE / "reports/metrics.json").read_text(encoding="utf-8"))
PREP = model.named_steps["prep"]
CLF = model.named_steps["clf"]
FEATURE_NAMES = [
    name.split("__", 1)[1] if "__" in name else name.split(" ", 1)[-1]
    for name in PREP.get_feature_names_out()
]
EXPLAINER = shap.TreeExplainer(CLF)

app = Flask(__name__)


def risk_label(prob: float) -> str:
    if prob >= THRESHOLD:
        return "High"
    if prob >= THRESHOLD / 2:
        return "Medium"
    return "Low"


def top_drivers(raw_row: pd.DataFrame, k: int = 3) -> list[dict]:
    """Return the k features that pushed this customer's score up or down the most."""
    X = add_features(raw_row)[FEATURES]
    shap_row = EXPLAINER.shap_values(PREP.transform(X))[0]
    order = np.argsort(-np.abs(shap_row))[:k]
    return [
        {"feature": FEATURE_NAMES[i], "impact": round(float(shap_row[i]), 3)}
        for i in order
    ]


@app.get("/")
def index():
    return render_template("index.html", choices=CHOICES, numeric=NUMERIC_INPUTS)


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    row = {key: payload[key] for key in list(CHOICES) + NUMERIC_INPUTS}
    for key in NUMERIC_INPUTS:
        row[key] = float(row[key])
    raw = pd.DataFrame([row])
    proba = float(model.predict_proba(add_features(raw)[FEATURES])[:, 1][0])
    return jsonify({
        "probability": round(proba, 3),
        "flagged": bool(proba >= THRESHOLD),
        "threshold": THRESHOLD,
        "risk": risk_label(proba),
        "drivers": top_drivers(raw),
    })


@app.post("/batch")
def batch():
    raw = pd.read_csv(request.files["file"])
    proba = model.predict_proba(add_features(raw)[FEATURES])[:, 1]
    ids = raw["customerID"] if "customerID" in raw.columns else raw.index.astype(str)
    out = pd.DataFrame({
        "customerID": ids.values,
        "churn_probability": np.round(proba, 3),
        "action": np.where(proba >= THRESHOLD, "Send retention offer", "No action"),
    }).sort_values("churn_probability", ascending=False)
    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_scores.csv"},
    )


@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS, threshold=THRESHOLD)


if __name__ == "__main__":
    app.run(debug=True)
