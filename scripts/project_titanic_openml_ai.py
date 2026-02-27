import argparse
import socket
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def can_connect(host: str, port: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    num_cols = x.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]

    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def evaluate_models(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    pre = build_preprocessor(x)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "avg_precision": "average_precision",
    }

    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=500, random_state=42, class_weight="balanced_subsample"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        score = cross_validate(pipe, x, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "roc_auc_mean": score["test_roc_auc"].mean(),
                "f1_mean": score["test_f1"].mean(),
                "avg_precision_mean": score["test_avg_precision"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proyecto IA: clasificacion Titanic usando OpenML."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Evita llamadas remotas a OpenML y usa fallback local.",
    )
    args = parser.parse_args()

    cache_dir = Path("data/external/openml_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = "OpenML Titanic (version 1)"
    try:
        if args.offline or not can_connect("api.openml.org"):
            raise RuntimeError("Sin conectividad a OpenML.")
        bunch = fetch_openml(name="titanic", version=1, as_frame=True, data_home=str(cache_dir))
        df = bunch.frame.copy()
        y = (df["survived"].astype(str) == "1").astype(int)
        x = df.drop(columns=["survived"])
    except Exception as exc:
        local = load_breast_cancer(as_frame=True)
        x = local.data
        y = local.target
        dataset_label = f"Fallback sklearn breast_cancer (sin red: {type(exc).__name__})"

    ranking = evaluate_models(x, y)
    best_name = ranking.iloc[0]["model"]

    model_map = {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=500, random_state=42, class_weight="balanced_subsample"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = Pipeline([("prep", build_preprocessor(x_train)), ("model", model_map[best_name])])
    pipe.fit(x_train, y_train)

    pred = pipe.predict(x_test)
    prob = pipe.predict_proba(x_test)[:, 1]
    roc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)
    report = classification_report(y_test, pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / "titanic_openml_model_ranking.csv"
    summary_path = args.output_dir / "titanic_openml_summary.md"
    ranking.to_csv(ranking_path, index=False)

    summary = [
        "# Titanic OpenML AI Project",
        "",
        f"- Dataset: {dataset_label}",
        f"- Mejor modelo CV: **{best_name}**",
        f"- ROC-AUC holdout: **{roc:.4f}**",
        f"- Average Precision holdout: **{ap:.4f}**",
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
