import argparse
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from model.model_utils import (
    ARTIFACT_DIR,
    ensure_dir,
    save_json,
    pack_eval_artifacts,
    make_metrics_table_md,
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
}


def train_and_save(force: bool = False) -> None:
    ensure_dir()

    if not force:
        ok = all((ARTIFACT_DIR / f).exists() for f in MODEL_FILES.values())
        ok = ok and (ARTIFACT_DIR / "metrics.json").exists()
        ok = ok and (ARTIFACT_DIR / "schema.json").exists()
        if ok:
            return

    # UCI dataset (via sklearn wrapper). Meets ≥500 rows and ≥12 features.
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    schema = {
        "dataset_name": "UCI Breast Cancer Wisconsin (Diagnostic) (via sklearn wrapper)",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_column": "target",
        "feature_columns": list(X.columns),
        "class_mapping": {"0": "malignant", "1": "benign"},
        "note": "Upload CSV must include all feature columns; optional target enables evaluation.",
    }
    save_json(ARTIFACT_DIR / "schema.json", schema)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6 required models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        ),
    }

    metrics_all = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, cm, report = pack_eval_artifacts(model, X_test, y_test)
        metrics_all[name] = metrics
        joblib.dump(model, ARTIFACT_DIR / MODEL_FILES[name])

    save_json(ARTIFACT_DIR / "metrics.json", metrics_all)

    md_table = make_metrics_table_md(metrics_all)
    (ARTIFACT_DIR / "metrics_table.md").write_text(md_table, encoding="utf-8")
    print(md_table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    train_and_save(force=args.force)


if __name__ == "__main__":
    main()
