import argparse

import matplotlib.pyplot as plt
import numpy as np


# Datos extraidos de las imagenes originales:
# matriz_confusion_test.png
# matriz_confusion_test_split_train.png
MATRIZ_TEST = np.array([
    [2555, 815],
    [2165, 1233],
])

MATRIZ_TEST_SPLIT_TRAIN = np.array([
    [3221, 139],
    [312, 16470],
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crea una imagen con dos matrices de confusion en porcentaje."
    )
    parser.add_argument(
        "--output",
        default="matrices_confusion_porcentaje.png",
        help="Nombre de la imagen final.",
    )
    parser.add_argument(
        "--output_test",
        default="matriz_confusion_test_porcentaje.png",
        help="Imagen individual para matriz_confusion_test.png.",
    )
    parser.add_argument(
        "--output_split",
        default="matriz_confusion_test_split_train_porcentaje.png",
        help="Imagen individual para matriz_confusion_test_split_train.png.",
    )
    return parser.parse_args()


def matriz_a_porcentaje(matriz):
    matriz = matriz.astype(float)
    total = matriz.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        porcentaje = np.divide(
            matriz,
            total,
            out=np.zeros_like(matriz),
            where=total != 0,
        ) * 100.0

    return porcentaje


def dibujar_matriz(ax, matriz):
    porcentaje = matriz_a_porcentaje(matriz)

    ax.imshow(porcentaje, cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"])
    ax.set_yticklabels(["real", "fake"])

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            texto = f"{int(matriz[i, j]):,}\n{porcentaje[i, j]:.1f}%"
            color = "white" if porcentaje[i, j] >= porcentaje.max() / 2 else "black"
            ax.text(
                j,
                i,
                texto,
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.set_aspect("equal")


def guardar_matriz(output, matriz):
    fig, ax = plt.subplots(figsize=(4, 4))
    dibujar_matriz(ax, matriz)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def guardar_imagen(output):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    dibujar_matriz(
        axes[0],
        MATRIZ_TEST,
    )
    dibujar_matriz(
        axes[1],
        MATRIZ_TEST_SPLIT_TRAIN,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    args = parse_args()
    guardar_imagen(args.output)
    guardar_matriz(args.output_test, MATRIZ_TEST)
    guardar_matriz(args.output_split, MATRIZ_TEST_SPLIT_TRAIN)

    print("Matriz test:")
    print(MATRIZ_TEST)
    print("Matriz test split train:")
    print(MATRIZ_TEST_SPLIT_TRAIN)
    print("Imagen guardada:", args.output)
    print("Imagen test guardada:", args.output_test)
    print("Imagen split guardada:", args.output_split)


if __name__ == "__main__":
    main()
