import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

ARTIFACT_DIR = Path(__file__).resolve().parent


def ensure_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_predict_proba(model, X):
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def pack_eval_artifacts(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = safe_predict_proba(model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)
    return metrics, cm, report


def make_metrics_table_md(metrics_dict: dict) -> str:
    headers = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    order = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)",
    ]

    def _fmt(x, nd=4):
        return f"{x:.{nd}f}"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for name in order:
        m = metrics_dict[name]
        row = [
            name,
            _fmt(m["accuracy"]),
            _fmt(m["auc"]),
            _fmt(m["precision"]),
            _fmt(m["recall"]),
            _fmt(m["f1"]),
            _fmt(m["mcc"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"
