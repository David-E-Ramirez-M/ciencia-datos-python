import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {suffix}")


def iqr_outliers(series: pd.Series) -> int:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA rapido y exportable para datasets tabulares."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("NTT/coffee_db.parquet"),
        help="Ruta a CSV o Parquet.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Cantidad de columnas numericas con mayor correlacion absoluta.",
    )
    args = parser.parse_args()

    df = load_dataframe(args.input)
    numeric = df.select_dtypes(include="number")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "eda_summary.md"

    missing_ratio = (df.isna().mean() * 100).sort_values(ascending=False)
    outlier_counts = {
        col: iqr_outliers(numeric[col].dropna()) for col in numeric.columns if not numeric[col].empty
    }

    corr_text = "No hay suficientes columnas numericas para correlacion."
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True).abs()
        corr_pairs = (
            corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
            .stack()
            .sort_values(ascending=False)
            .head(args.top_n)
        )
        corr_text = corr_pairs.to_string()

    lines = [
        "# EDA Summary",
        "",
        f"- Fuente: `{args.input}`",
        f"- Filas: **{len(df):,}**",
        f"- Columnas: **{df.shape[1]}**",
        "",
        "## Missing Values (%)",
        missing_ratio.to_string(),
        "",
        "## Outliers IQR (conteo por columna numerica)",
        pd.Series(outlier_counts).sort_values(ascending=False).to_string()
        if outlier_counts
        else "No hay columnas numericas.",
        "",
        "## Top Correlaciones Absolutas",
        corr_text,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"EDA generado: {output_path}")


if __name__ == "__main__":
    main()
