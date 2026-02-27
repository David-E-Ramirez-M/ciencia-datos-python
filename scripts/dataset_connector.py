import argparse
import json
import socket
import subprocess
from pathlib import Path

import pandas as pd
import requests
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine


SKLEARN_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

OPENML_DATASETS = {
    "titanic": {"name": "titanic", "version": 1},
    "adult": {"name": "adult", "version": 2},
    "house_prices": {"name": "house_prices", "version": 1},
}

URL_DATASETS = {
    "wine_quality_red": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "sep": ";",
    },
    "wine_quality_white": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "sep": ";",
    },
}


def can_connect(host: str, port: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def save_dataframe(df: pd.DataFrame, output_dir: Path, filename: str, metadata: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{filename}.csv"
    meta_path = output_dir / f"{filename}.metadata.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    return csv_path


def load_sklearn_dataset(name: str) -> tuple[pd.DataFrame, dict]:
    if name not in SKLEARN_DATASETS:
        raise ValueError(f"Dataset sklearn no soportado: {name}")
    bunch = SKLEARN_DATASETS[name](as_frame=True)
    df = bunch.frame.copy()
    meta = {
        "source": "sklearn",
        "dataset": name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target_names": list(getattr(bunch, "target_names", [])),
    }
    return df, meta


def load_openml_dataset(name: str) -> tuple[pd.DataFrame, dict]:
    if name not in OPENML_DATASETS:
        raise ValueError(f"Dataset OpenML no soportado: {name}")
    cache_dir = Path("data/external/openml_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = OPENML_DATASETS[name]
    bunch = fetch_openml(
        name=cfg["name"], version=cfg["version"], as_frame=True, data_home=str(cache_dir)
    )
    df = bunch.frame.copy()
    meta = {
        "source": "openml",
        "dataset": name,
        "openml_name": cfg["name"],
        "version": cfg["version"],
        "target_column": bunch.target.name if hasattr(bunch.target, "name") else None,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }
    return df, meta


def load_url_dataset(name: str) -> tuple[pd.DataFrame, dict]:
    if name not in URL_DATASETS:
        raise ValueError(f"Dataset URL no soportado: {name}")
    cfg = URL_DATASETS[name]
    response = requests.get(cfg["url"], timeout=60)
    response.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(response.text), sep=cfg["sep"])
    meta = {
        "source": "url",
        "dataset": name,
        "url": cfg["url"],
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }
    return df, meta


def download_kaggle_dataset(dataset_slug: str, output_dir: Path, unzip: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", dataset_slug, "-p", str(output_dir)]
    if unzip:
        cmd.append("--unzip")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conecta y descarga datasets para proyectos de IA."
    )
    parser.add_argument(
        "--source",
        choices=["sklearn", "openml", "url", "kaggle"],
        required=True,
        help="Origen del dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="iris",
        help="Nombre en catalogo local (sklearn/openml/url).",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="",
        help="Slug Kaggle, ejemplo: yasserh/titanic-dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directorio destino.",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Descomprimir al descargar de Kaggle.",
    )
    parser.add_argument(
        "--fallback-sklearn",
        type=str,
        default="",
        help="Dataset sklearn de respaldo (ejemplo: iris) si falla red.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Evita descargas remotas (OpenML/URL/Kaggle) y usa fallback si aplica.",
    )
    args = parser.parse_args()

    if args.source == "kaggle":
        if args.offline:
            raise RuntimeError("Modo offline activo: Kaggle deshabilitado.")
        if not args.kaggle_dataset:
            raise ValueError("Debes indicar --kaggle-dataset para source=kaggle.")
        download_kaggle_dataset(args.kaggle_dataset, args.output_dir, args.unzip)
        print(f"Dataset Kaggle descargado en: {args.output_dir}")
        return

    output_name = args.name
    try:
        if args.source == "sklearn":
            df, meta = load_sklearn_dataset(args.name)
        elif args.source == "openml":
            if args.offline or not can_connect("api.openml.org"):
                raise RuntimeError("Sin conectividad a OpenML.")
            df, meta = load_openml_dataset(args.name)
        else:
            if args.offline or not can_connect("archive.ics.uci.edu"):
                raise RuntimeError("Sin conectividad a UCI.")
            df, meta = load_url_dataset(args.name)
    except Exception as exc:
        if args.fallback_sklearn:
            df, meta = load_sklearn_dataset(args.fallback_sklearn)
            meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            meta["original_source"] = args.source
            meta["original_name"] = args.name
            output_name = f"{args.name}_fallback_{args.fallback_sklearn}"
            print(
                "Fallo la descarga remota. "
                f"Se uso fallback sklearn='{args.fallback_sklearn}'."
            )
        else:
            raise

    output_path = save_dataframe(df, args.output_dir, output_name, meta)
    print(f"Dataset guardado en: {output_path}")
    print(f"Filas={df.shape[0]}, Columnas={df.shape[1]}")


if __name__ == "__main__":
    main()
