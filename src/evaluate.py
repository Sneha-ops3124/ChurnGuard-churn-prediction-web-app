import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from src.features import FEATURES, load_training_data

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = BASE / "reports/figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_training_data(DATA_PATH)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = joblib.load(BASE / "models/churn_model.pkl")
    threshold = json.loads((BASE / "models/threshold.json").read_text(encoding="utf-8"))["threshold"]
    proba = model.predict_proba(X_test[FEATURES])[:, 1]

    # ROC, PR, and confusion matrix side by side
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax[0])
    ax[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax[1])
    ax[1].set_title("Precision-Recall curve")
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        (proba >= threshold).astype(int),
        ax=ax[2],
        display_labels=["Stayed", "Churned"],
    )
    ax[2].set_title(f"Confusion matrix (threshold={threshold})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_evaluation.png", dpi=150)
    plt.close(fig)

    # SHAP summary for the deployed tree model
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    X_t = prep.transform(X_test[FEATURES])
    names = [n.split("__", 1)[1] if "__" in n else n.split(" ", 1)[-1] for n in prep.get_feature_names_out()]
    shap_values = shap.TreeExplainer(clf).shap_values(X_t)
    shap.summary_plot(shap_values, X_t, feature_names=names, max_display=12, show=False)
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved plots to", FIG_DIR)


if __name__ == "__main__":
    main()
