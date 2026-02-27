import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def resolve_input_path(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]

    if path.is_absolute():
        return path
    if path.exists():
        return path

    candidate = repo_root / path
    if candidate.exists():
        return candidate

    if path.name == "coffee_db.parquet":
        fallback = repo_root / "MiniProyecto" / "coffee_db.parquet"
        if fallback.exists():
            return fallback

    return candidate


def load_series(input_path: Path, column: str) -> pd.Series:
    input_path = resolve_input_path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Formato no soportado: {suffix}")

    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el dataset.")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        raise ValueError("La columna seleccionada no tiene datos numericos validos.")
    return series


def run_diagnostics(series: pd.Series) -> dict:
    values = series.to_numpy()
    n = len(values)

    shapiro = stats.shapiro(values)
    mean = values.mean()
    std = values.std(ddof=1)
    standardized = (values - mean) / std if std > 0 else np.zeros_like(values)
    ks = stats.kstest(standardized, "norm")
    anderson = stats.anderson(values, dist="norm")

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "shapiro_stat": shapiro.statistic,
        "shapiro_p": shapiro.pvalue,
        "ks_stat": ks.statistic,
        "ks_p": ks.pvalue,
        "anderson_stat": anderson.statistic,
        "anderson_critical_5pct": anderson.critical_values[2],
    }


def save_plots(series: pd.Series, column: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_path = output_dir / f"{column}_hist.png"
    qq_path = output_dir / f"{column}_qqplot.png"

    plt.figure(figsize=(7, 4))
    plt.hist(series, bins=20, edgecolor="black", alpha=0.8)
    plt.title(f"Histograma - {column}")
    plt.xlabel(column)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    fig = sm.qqplot(series, line="s")
    fig.suptitle(f"QQ-plot - {column}")
    fig.tight_layout()
    fig.savefig(qq_path, dpi=150)
    plt.close(fig)
    return hist_path, qq_path


def recommendation(n: int, shapiro_p: float, ks_p: float, anderson_stat: float, anderson_critical: float) -> str:
    passes_tests = shapiro_p > 0.05 and ks_p > 0.05 and anderson_stat < anderson_critical
    if passes_tests:
        return "Normalidad razonable para metodos parametricos."
    if n >= 30:
        return (
            "Hay desviaciones de normalidad. Si usas inferencia parametrica, valida residuos, "
            "considera transformaciones (log/Yeo-Johnson) o metodos robustos."
        )
    return (
        "Muestra pequena con evidencia de no normalidad. Prioriza pruebas no parametricas "
        "o transformaciones justificadas."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostico de normalidad: Shapiro, KS, Anderson y QQ-plot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("MiniProyecto/coffee_db.parquet"),
        help="Ruta a dataset (CSV/Parquet).",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="Total_domestic_consumption",
        help="Columna numerica a evaluar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directorio para guardar figuras.",
    )
    args = parser.parse_args()

    series = load_series(args.input, args.column)
    diag = run_diagnostics(series)
    hist_path, qq_path = save_plots(series, args.column, args.output_dir)

    summary = [
        "# Normality Diagnostics",
        "",
        f"- Source: `{args.input}`",
        f"- Column: `{args.column}`",
        f"- N: **{diag['n']}**",
        f"- Mean: **{diag['mean']:.4f}**",
        f"- Std: **{diag['std']:.4f}**",
        "",
        "## Tests",
        f"- Shapiro-Wilk: stat={diag['shapiro_stat']:.4f}, p={diag['shapiro_p']:.6f}",
        f"- Kolmogorov-Smirnov: stat={diag['ks_stat']:.4f}, p={diag['ks_p']:.6f}",
        (
            "- Anderson-Darling: "
            f"stat={diag['anderson_stat']:.4f}, critical@5%={diag['anderson_critical_5pct']:.4f}"
        ),
        "",
        "## Recommendation",
        recommendation(
            diag["n"],
            diag["shapiro_p"],
            diag["ks_p"],
            diag["anderson_stat"],
            diag["anderson_critical_5pct"],
        ),
        "",
        "## Figures",
        f"- Histogram: `{hist_path}`",
        f"- QQ-plot: `{qq_path}`",
    ]

    out_path = Path("reports/normality_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"Reporte generado: {out_path}")
    print(f"Figuras: {hist_path}, {qq_path}")


if __name__ == "__main__":
    main()
