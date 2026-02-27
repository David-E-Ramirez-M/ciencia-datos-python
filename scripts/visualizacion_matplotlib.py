import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera una visualizacion comparando senal real vs ruido."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/seno_ruido.png"),
        help="Ruta de salida de la figura.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad."
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, alpha=0.7, label="Datos con ruido")
    plt.plot(x, np.sin(x), color="red", label="Funcion objetivo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Ejemplo de visualizacion de datos")
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(f"Figura guardada en: {args.output}")


if __name__ == "__main__":
    main()
