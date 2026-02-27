from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_dir() -> Path:
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_banner(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.axis("off")

    ax.text(
        0.02,
        0.78,
        "Data Science Portfolio",
        fontsize=28,
        fontweight="bold",
        color="#F8FAFC",
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.56,
        "Modelado, estadistica aplicada y storytelling para decisiones de negocio.",
        fontsize=12,
        color="#CBD5E1",
        transform=ax.transAxes,
    )

    cards = [
        ("13+", "notebooks tecnicos"),
        ("4", "pipelines scriptables"),
        ("3", "datasets conectables"),
    ]
    x_positions = [0.02, 0.30, 0.56]
    for (value, label), x in zip(cards, x_positions):
        ax.add_patch(
            plt.Rectangle(
                (x, 0.08),
                0.22,
                0.34,
                color="#1E293B",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(x + 0.03, 0.27, value, fontsize=20, fontweight="bold", color="#38BDF8", transform=ax.transAxes)
        ax.text(x + 0.03, 0.13, label, fontsize=11, color="#E2E8F0", transform=ax.transAxes)

    out = output_dir / "readme_banner.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_model_snapshot(output_dir: Path) -> Path:
    iris = pd.read_csv("reports/iris_model_ranking.csv")
    titanic = pd.read_csv("reports/titanic_openml_model_ranking.csv")
    wine = pd.read_csv("reports/wine_quality_model_ranking.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    axes[0].barh(iris["model"], iris["f1_macro_mean"], color="#2563EB")
    axes[0].set_title("Iris - F1 Macro")
    axes[0].set_xlabel("Score")
    axes[0].invert_yaxis()

    axes[1].barh(titanic["model"], titanic["roc_auc_mean"], color="#0EA5E9")
    axes[1].set_title("Titanic/Fallback - ROC-AUC")
    axes[1].set_xlabel("Score")
    axes[1].invert_yaxis()

    axes[2].barh(wine["model"], wine["rmse_mean"], color="#14B8A6")
    axes[2].set_title("Wine/Fallback - RMSE (menor mejor)")
    axes[2].set_xlabel("Error")
    axes[2].invert_yaxis()

    fig.suptitle("Model Snapshot", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "model_snapshot.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def build_miniproyecto_trend(output_dir: Path) -> Path:
    df = pd.read_parquet("MiniProyecto/coffee_db.parquet")
    year_cols = [c for c in df.columns if "/" in c]
    top = df.nlargest(5, "Total_domestic_consumption")

    long_df = top.melt(
        id_vars=["Country", "Total_domestic_consumption"],
        value_vars=year_cols,
        var_name="period",
        value_name="consumption",
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for country, part in long_df.groupby("Country"):
        ax.plot(part["period"], part["consumption"], marker="o", linewidth=1.7, label=country)

    ax.set_title("MiniProyecto: Top 5 paises por consumo de cafe")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Consumo")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    out = output_dir / "miniproyecto_top5_trend.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    output_dir = ensure_output_dir()
    banner = build_banner(output_dir)
    snapshot = build_model_snapshot(output_dir)
    trend = build_miniproyecto_trend(output_dir)
    print(f"Generado: {banner}")
    print(f"Generado: {snapshot}")
    print(f"Generado: {trend}")


if __name__ == "__main__":
    main()
