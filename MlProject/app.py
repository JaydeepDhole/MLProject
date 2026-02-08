import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from model.train_models import train_and_save,MODEL_FILES


ARTIFACT_DIR = Path(__file__).resolve().parent / "model"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_predict_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


st.set_page_config(page_title="ML Project ", layout="wide")
st.title("ML Project — Classification Models (Streamlit)")

schema_path = ARTIFACT_DIR / "schema.json"
metrics_path = ARTIFACT_DIR / "metrics.json"

# One-click training (nice for Streamlit Cloud)
if (not schema_path.exists()) or (not metrics_path.exists()):
    st.warning("First-time setup: artifacts not found. Click to train models and generate metrics.")
    if st.button("Train models now"):
        with st.spinner("Training..."):
            train_and_save(force=True)
        st.success("Training complete. Reloading…")
        st.rerun()
    st.stop()

schema = load_json(schema_path)
pre_metrics = load_json(metrics_path)

st.markdown(
    f"""
**Dataset:** {schema['dataset_name']}  
**Rows:** {schema['n_rows']} | **Features:** {schema['n_features']}  
"""
)

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Select Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    model = joblib.load(ARTIFACT_DIR / MODEL_FILES[model_name])

with right:
    st.subheader("2) Metrics (Hold-out split)")
    m = pre_metrics[model_name]
    st.dataframe(pd.DataFrame([{
        "Accuracy": m["accuracy"],
        "AUC": m["auc"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "MCC": m["mcc"],
    }]), use_container_width=True)

st.divider()
st.subheader("3) Upload CSV (Test Data)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
with st.expander("Required feature columns"):
    st.write(schema["feature_columns"])

if uploaded is None:
    st.info("Use sample_test.csv / sample_test_no_target.csv, or upload your own test CSV.")
    st.stop()

df = pd.read_csv(uploaded)
st.dataframe(df.head(10), use_container_width=True)

feature_cols = schema["feature_columns"]
target_col = schema.get("target_column", "target")

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Missing {len(missing)} feature columns. Example: {missing[:5]}")
    st.stop()

X = df[feature_cols]
y_pred = model.predict(X)
y_proba = safe_predict_proba(model, X)

out = df.copy()
out["predicted"] = y_pred
out["probability_positive_class"] = y_proba

st.subheader("4) Predictions")
st.dataframe(out.head(25), use_container_width=True)

st.download_button(
    "Download predictions CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)

if target_col in df.columns:
    st.subheader("5) Confusion Matrix + Report")
    y_true = df[target_col].astype(int)
    metrics_live = compute_metrics(y_true, y_pred, y_proba)
    st.write(pd.DataFrame([metrics_live]))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax, values_format="d")
    st.pyplot(fig)

    st.code(classification_report(y_true, y_pred, zero_division=0))
else:
    st.warning("No target column found — predictions shown only.")
