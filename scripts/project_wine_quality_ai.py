import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WINE_RED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)


def download_wine_dataset() -> pd.DataFrame:
    response = requests.get(WINE_RED_URL, timeout=60)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep=";")


def evaluate_models(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

    models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=500, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    for name, model in models.items():
        score = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "rmse_mean": -score["test_rmse"].mean(),
                "mae_mean": -score["test_mae"].mean(),
                "r2_mean": score["test_r2"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse_mean")


def main() -> None:
    parser = argparse.ArgumentParser(description="Proyecto IA: regresion con Wine Quality (UCI).")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--save-raw", action="store_true", help="Guardar copia en data/raw.")
    args = parser.parse_args()

    dataset_label = f"UCI Wine Quality ({WINE_RED_URL})"
    try:
        df = download_wine_dataset()
        y = df["quality"]
        x = df.drop(columns=["quality"])
    except Exception as exc:
        local = load_diabetes(as_frame=True)
        x = local.data
        y = local.target
        df = pd.concat([x, y.rename("target")], axis=1)
        dataset_label = f"Fallback sklearn diabetes (sin red: {type(exc).__name__})"

    ranking = evaluate_models(x, y)
    best_name = ranking.iloc[0]["model"]

    model_map = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=500, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    model = model_map[best_name]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    rmse = root_mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / "wine_quality_model_ranking.csv"
    summary_path = args.output_dir / "wine_quality_summary.md"
    ranking.to_csv(ranking_path, index=False)

    if args.save_raw:
        raw_path = Path("data/raw/wine_quality_red.csv")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False, encoding="utf-8")

    summary = [
        "# Wine Quality AI Project",
        "",
        f"- Dataset source: `{dataset_label}`",
        f"- Mejor modelo CV (RMSE): **{best_name}**",
        f"- RMSE holdout: **{rmse:.4f}**",
        f"- MAE holdout: **{mae:.4f}**",
        f"- R2 holdout: **{r2:.4f}**",
        "",
        f"Ranking CSV: `{ranking_path}`",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print(f"Resumen: {summary_path}")
    print(f"Ranking: {ranking_path}")


if __name__ == "__main__":
    main()
