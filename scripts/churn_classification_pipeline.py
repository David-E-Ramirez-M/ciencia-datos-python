import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(csv_path: Path | None, target: str) -> tuple[pd.DataFrame, pd.Series, str]:
    if csv_path is None:
        data = load_breast_cancer(as_frame=True)
        x = data.data
        y = data.target
        return x, y, "demo_breast_cancer"

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"La columna target '{target}' no existe.")
    y = df[target]
    x = df.drop(columns=[target])
    return x, y, csv_path.stem


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    num_cols = x.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def evaluate_models(x: pd.DataFrame, y: pd.Series, n_jobs: int) -> pd.DataFrame:
    preprocessor = build_preprocessor(x)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "avg_precision": "average_precision",
    }

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced_subsample"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        scores = cross_validate(pipe, x, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        rows.append(
            {
                "model": name,
                "roc_auc_mean": scores["test_roc_auc"].mean(),
                "f1_mean": scores["test_f1"].mean(),
                "precision_mean": scores["test_precision"].mean(),
                "recall_mean": scores["test_recall"].mean(),
                "avg_precision_mean": scores["test_avg_precision"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def fit_best_model(
    x: pd.DataFrame, y: pd.Series, best_model_name: str
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    preprocessor = build_preprocessor(x)
    model_map = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced_subsample"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    model = model_map[best_model_name]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(x_train, y_train)
    return pipe, x_test, y_test, pipe.predict(x_test)


def save_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, model_name: str, dataset_name: str
) -> Path:
    output_path = Path("reports/figures") / f"cm_{dataset_name}_{model_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de clasificacion binaria end-to-end con CV y metricas."
    )
    parser.add_argument("--csv", type=Path, default=None, help="Ruta a dataset CSV.")
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Columna objetivo cuando se use --csv.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Procesos para cross-validation. Usa 1 en entornos restringidos.",
    )
    args = parser.parse_args()

    x, y, dataset_name = load_data(args.csv, args.target)
    results = evaluate_models(x, y, args.n_jobs)
    best_model = results.iloc[0]["model"]

    model, x_test, y_test, y_pred = fit_best_model(x, y, best_model)
    y_score = model.predict_proba(x_test)[:, 1]

    report = classification_report(y_test, y_pred)
    holdout_roc = roc_auc_score(y_test, y_score)
    holdout_ap = average_precision_score(y_test, y_score)
    cm_path = save_confusion_matrix(y_test, y_pred, best_model, dataset_name)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = reports_dir / f"{dataset_name}_model_comparison.csv"
    results.to_csv(comparison_path, index=False)

    summary_path = reports_dir / f"{dataset_name}_classification_summary.md"
    summary_lines = [
        "# Clasificacion End-to-End",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Mejor modelo por ROC-AUC CV: **{best_model}**",
        f"- ROC-AUC holdout: **{holdout_roc:.4f}**",
        f"- Average Precision holdout: **{holdout_ap:.4f}**",
        "",
        "## Classification Report (holdout)",
        "```text",
        report,
        "```",
        "",
        f"Confusion matrix: `{cm_path}`",
        f"Tabla comparativa CV: `{comparison_path}`",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Resumen generado: {summary_path}")
    print(f"Comparacion de modelos: {comparison_path}")
    print(f"Matriz de confusion: {cm_path}")


if __name__ == "__main__":
    main()
