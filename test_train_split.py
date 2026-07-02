import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from ImputData import SurFakeDataset
from Modetrain import MobileNetV2_SurFake
from train import metricas, split_por_video


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():

    parser = argparse.ArgumentParser(
        description="Testea el split test de train.py sin entrenar."
    )

    parser.add_argument(
        "--data_dir",
        default="data_completa",
        help="Dataset usado por train.py."
    )

    parser.add_argument(
        "--modelo_path",
        default="modelo_actual.pth",
        help="Checkpoint del modelo entrenado."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Cantidad de muestras por batch."
    )

    parser.add_argument(
        "--csv",
        default="resultados_test_split_train.csv",
        help="Archivo CSV donde se guardan las metricas."
    )

    parser.add_argument(
        "--matriz_img",
        default="matriz_confusion_test_split_train.png",
        help="Imagen PNG donde se guarda la matriz de confusion."
    )

    parser.add_argument(
        "--metricas_grupos_csv",
        default="metricas_grupos_test_split_train.csv",
        help="CSV donde se guardan metricas por clase y por tecnica fake."
    )

    parser.add_argument(
        "--resultados_dir",
        default=None,
        help="Carpeta base donde se guardan los resultados del test."
    )

    parser.add_argument(
        "--nombre_resultado",
        default=None,
        help="Nombre de la carpeta nueva para este resultado."
    )

    return parser.parse_args()


def preparar_rutas_resultados(args):

    if args.resultados_dir is None and args.nombre_resultado is None:
        return args.csv, args.matriz_img, args.metricas_grupos_csv

    resultados_dir = args.resultados_dir or "resultados_tests"
    nombre_resultado = args.nombre_resultado or "test_split_resultado"
    carpeta = os.path.join(resultados_dir, nombre_resultado)

    os.makedirs(carpeta, exist_ok=True)

    csv_path = os.path.join(carpeta, os.path.basename(args.csv))
    matriz_path = os.path.join(carpeta, os.path.basename(args.matriz_img))
    grupos_path = os.path.join(carpeta, os.path.basename(args.metricas_grupos_csv))

    return csv_path, matriz_path, grupos_path


def cargar_modelo(modelo_path):

    modelo = MobileNetV2_SurFake(
        num_classes=2,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(modelo_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        modelo.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "desconocida")

    else:
        modelo.load_state_dict(checkpoint)
        epoch = "desconocida"

    return modelo, epoch


def evaluar_con_matriz(modelo, loader, criterion, nombre="test"):

    modelo.eval()

    total_loss = 0.0

    y_true = []
    y_pred = []
    y_prob = []

    barra = tqdm(loader, total=len(loader), desc=nombre, leave=False)

    with torch.no_grad():

        for x, y in barra:

            x = x.to(device)
            y = y.to(device)

            salida = modelo(x)

            loss = criterion(salida, y)

            probs = torch.softmax(salida, dim=1)[:, 1]
            preds = salida.argmax(dim=1)

            total_loss += loss.item() * x.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            barra.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gpu": torch.cuda.is_available()
            })

    avg_loss = total_loss / len(loader.dataset)

    acc, f1, auc = metricas(y_true, y_pred, y_prob)
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return avg_loss, acc, f1, auc, matriz, y_true, y_pred, y_prob


def guardar_csv(csv_path, data_dir, modelo_path, epoch, loss, acc, f1, auc, matriz):

    tn, fp, fn, tp = matriz[0, 0], matriz[0, 1], matriz[1, 0], matriz[1, 1]

    with open(csv_path, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "data_dir",
            "modelo_path",
            "epoch",
            "test_loss",
            "test_acc",
            "test_f1",
            "test_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "confusion_matrix"
        ])

        writer.writerow([
            data_dir,
            modelo_path,
            epoch,
            loss,
            acc,
            f1,
            auc,
            tn,
            fp,
            fn,
            tp,
            f"[[{tn}, {fp}], [{fn}, {tp}]]"
        ])


def get_grupo_muestra(rgb_path):

    path = rgb_path.replace("\\", "/")
    parts = path.split("/")

    if "real" in parts:
        return "real"

    if "fake" in parts:
        idx = parts.index("fake")
        tecnica = parts[idx + 1] if idx + 1 < len(parts) else "fake_desconocido"
        return f"fake_{tecnica}"

    return "desconocido"


def metricas_de_indices(nombre, indices, y_true, y_pred, y_prob):

    yt = [y_true[i] for i in indices]
    yp = [y_pred[i] for i in indices]
    ypr = [y_prob[i] for i in indices]

    acc, f1, auc = metricas(yt, yp, ypr)
    matriz = confusion_matrix(yt, yp, labels=[0, 1])

    return {
        "grupo": nombre,
        "n_muestras": len(indices),
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "tn": int(matriz[0, 0]),
        "fp": int(matriz[0, 1]),
        "fn": int(matriz[1, 0]),
        "tp": int(matriz[1, 1])
    }


def guardar_metricas_grupos(csv_path, test_dataset, y_true, y_pred, y_prob):

    grupos = {
        "general": list(range(len(y_true))),
        "real": [],
        "fake": []
    }

    for pos, idx in enumerate(test_dataset.indices):
        rgb_path, _, label = test_dataset.dataset.samples[idx]

        if label == 0:
            grupos["real"].append(pos)

        else:
            grupos["fake"].append(pos)
            grupo_fake = get_grupo_muestra(rgb_path)
            grupos.setdefault(grupo_fake, []).append(pos)

    filas = [
        metricas_de_indices(nombre, indices, y_true, y_pred, y_prob)
        for nombre, indices in grupos.items()
        if len(indices) > 0
    ]

    filas_fake_tecnicas = [
        fila for fila in filas
        if fila["grupo"].startswith("fake_")
    ]

    if filas_fake_tecnicas:
        filas.append({
            "grupo": "promedio_fake_tecnicas",
            "n_muestras": sum(fila["n_muestras"] for fila in filas_fake_tecnicas),
            "acc": sum(fila["acc"] for fila in filas_fake_tecnicas) / len(filas_fake_tecnicas),
            "f1": sum(fila["f1"] for fila in filas_fake_tecnicas) / len(filas_fake_tecnicas),
            "auc": sum(fila["auc"] for fila in filas_fake_tecnicas) / len(filas_fake_tecnicas),
            "tn": sum(fila["tn"] for fila in filas_fake_tecnicas),
            "fp": sum(fila["fp"] for fila in filas_fake_tecnicas),
            "fn": sum(fila["fn"] for fila in filas_fake_tecnicas),
            "tp": sum(fila["tp"] for fila in filas_fake_tecnicas)
        })

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "grupo",
                "n_muestras",
                "acc",
                "f1",
                "auc",
                "tn",
                "fp",
                "fn",
                "tp"
            ]
        )

        writer.writeheader()
        writer.writerows(filas)

    return filas


def guardar_matriz_img(img_path, matriz, titulo):

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matriz, cmap="Blues")

    ax.set_title(titulo)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["real 0", "fake 1"])
    ax.set_yticklabels(["real 0", "fake 1"])

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            color = "white" if matriz[i, j] > matriz.max() / 2 else "black"
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(img_path, dpi=200)
    plt.close(fig)


def main():

    args = parse_args()
    csv_path, matriz_path, grupos_path = preparar_rutas_resultados(args)

    print("device:", device)

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    else:
        print("cuda no disponible")

    dataset = SurFakeDataset(args.data_dir)

    _, _, test_dataset = split_por_video(dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    modelo, epoch = cargar_modelo(args.modelo_path)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_f1, test_auc, matriz, y_true, y_pred, y_prob = evaluar_con_matriz(
        modelo,
        test_loader,
        criterion,
        nombre="test_split_train"
    )

    guardar_csv(
        csv_path,
        args.data_dir,
        args.modelo_path,
        epoch,
        test_loss,
        test_acc,
        test_f1,
        test_auc,
        matriz
    )

    filas_grupos = guardar_metricas_grupos(
        grupos_path,
        test_dataset,
        y_true,
        y_pred,
        y_prob
    )

    guardar_matriz_img(
        matriz_path,
        matriz,
        "Matriz de confusion - test split train.py"
    )

    print("\ntest final del split de train.py")
    print("epoch checkpoint:", epoch)
    print(f"test loss: {test_loss:.4f}")
    print(f"test acc : {test_acc:.4f}")
    print(f"test f1  : {test_f1:.4f}")
    print(f"test auc : {test_auc:.4f}")
    print("\nmatriz confusion")
    print("filas: true real=0, true fake=1")
    print("cols : pred real=0, pred fake=1")
    print(matriz)
    print("\nmetricas por grupo")

    for fila in filas_grupos:
        print(
            fila["grupo"],
            "n:", fila["n_muestras"],
            "acc:", f"{fila['acc']:.4f}",
            "f1:", f"{fila['f1']:.4f}",
            "auc:", f"{fila['auc']:.4f}"
        )

    print("csv:", csv_path)
    print("metricas grupos csv:", grupos_path)
    print("matriz imagen:", matriz_path)


if __name__ == "__main__":
    main()
