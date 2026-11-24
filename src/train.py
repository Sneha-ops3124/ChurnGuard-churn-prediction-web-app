import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.features import build_preprocessor, load_training_data

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Business assumptions. Change these to match your own story and explain them in the README.
OFFER_COST = 20        # USD spent on each retention offer
CLV_LOST = 300         # USD lifetime value lost when a customer churns
SUCCESS_RATE = 0.5     # share of contacted churners who stay because of the offer
DEPLOYED_MODEL = "XGBoost"  # the web app uses TreeExplainer, which works with tree models


def profit(y_true, proba, threshold):
    """Simulated profit: every flagged customer gets an offer, and saved churners add value."""
    y_true = np.asarray(y_true)
    flagged = proba >= threshold
    saved = np.sum(flagged & (y_true == 1)) * SUCCESS_RATE * CLV_LOST
    spent = np.sum(flagged) * OFFER_COST
    return float(saved - spent)


def best_threshold(y_true, proba):
    grid = np.linspace(0.05, 0.95, 91)
    profits = [profit(y_true, proba, t) for t in grid]
    idx = int(np.argmax(profits))
    return float(grid[idx]), profits[idx]


def make_models(scale_pos_weight):
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def main():
    X, y = load_training_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    models = make_models(scale_pos_weight=neg / pos)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results, fitted = {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])

        # 1) Out-of-fold probabilities on TRAINING data only
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold, _ = best_threshold(y_train, oof)
        oof_pr_auc = average_precision_score(y_train, oof)

        # 2) Fit on all training data, then evaluate on the untouched test set
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        results[name] = {
            "threshold": round(threshold, 2),
            "cv_pr_auc": round(oof_pr_auc, 3),
            "precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, pred, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 3),
            "pr_auc": round(average_precision_score(y_test, proba), 3),
            "profit_test": round(profit(y_test, proba, threshold), 2),
        }
        fitted[name] = pipe
        print(
            f"{name:20s} CV PR-AUC={oof_pr_auc:.3f} "
            f"test ROC-AUC={results[name]['roc_auc']} "
            f"profit={results[name]['profit_test']}"
        )

    # Save artifacts
    (BASE / "models").mkdir(exist_ok=True)
    (BASE / "reports").mkdir(exist_ok=True)
    joblib.dump(fitted[DEPLOYED_MODEL], BASE / "models/churn_model.pkl")
    (BASE / "models/threshold.json").write_text(
        json.dumps(
            {
                "model": DEPLOYED_MODEL,
                "threshold": results[DEPLOYED_MODEL]["threshold"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (BASE / "reports/metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print("Saved models/churn_model.pkl, models/threshold.json, reports/metrics.json")


if __name__ == "__main__":
    main()
