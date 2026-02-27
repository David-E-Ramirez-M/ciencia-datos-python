import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def evaluate_models(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    models = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
        ),
        "svm_rbf": Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=42),
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "f1_macro_mean": scores["test_f1_macro"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Proyecto IA: clasificacion Iris end-to-end.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directorio de salida para reportes.",
    )
    args = parser.parse_args()

    bunch = load_iris(as_frame=True)
    x = bunch.data
    y = bunch.target

    ranking = evaluate_models(x, y)
    best_name = ranking.iloc[0]["model"]

    model_map = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
        ),
        "svm_rbf": Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=42),
    }
    model = model_map[best_name]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=bunch.target_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / "iris_model_ranking.csv"
    summary_path = args.output_dir / "iris_project_summary.md"

    ranking.to_csv(ranking_path, index=False)
    summary = [
        "# Iris AI Project",
        "",
        f"- Mejor modelo CV: **{best_name}**",
        f"- Accuracy holdout: **{acc:.4f}**",
        f"- F1 macro holdout: **{f1:.4f}**",
        "",
        "## Classification Report",
        "```text",
        report,
        "```",
        "",
        f"Ranking CSV: `{ranking_path}`",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print(f"Resumen: {summary_path}")
    print(f"Ranking: {ranking_path}")


if __name__ == "__main__":
    main()
