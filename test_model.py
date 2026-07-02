import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from ImputData import SurFakeDataset
from Modetrain import MobileNetV2_SurFake


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():

    parser = argparse.ArgumentParser(
        description="Testea un modelo entrenado sin volver a entrenar."
    )

    parser.add_argument(
        "--data_dir",
        default="data_completa2/test",
        help="Carpeta que contiene real, fake, real_gsd y fake_gsd."
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
        default="resultados_test.csv",
        help="Archivo CSV donde se guardan las metricas del test."
    )

    parser.add_argument(
        "--matriz_img",
        default="matriz_confusion_test.png",
        help="Imagen PNG donde se guarda la matriz de confusion."
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
        return args.csv, args.matriz_img

    resultados_dir = args.resultados_dir or "resultados_tests"
    nombre_resultado = args.nombre_resultado or "test_resultado"
    carpeta = os.path.join(resultados_dir, nombre_resultado)

    os.makedirs(carpeta, exist_ok=True)

    csv_path = os.path.join(carpeta, os.path.basename(args.csv))
    matriz_path = os.path.join(carpeta, os.path.basename(args.matriz_img))

    return csv_path, matriz_path


def metricas(y_true, y_pred, y_prob):

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)

    except ValueError:
        auc = 0.0

    return acc, f1, auc


def evaluar(modelo, loader, criterion, nombre="test"):

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

    return avg_loss, acc, f1, auc, matriz


def cargar_modelo(modelo_path):

    modelo = MobileNetV2_SurFake(
        num_classes=2,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(modelo_path, map_location=device)

    checkpoint_info = {}

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        modelo.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "desconocida")
        checkpoint_info = {
            "epoch": epoch,
            "train_loss": checkpoint.get("train_loss", ""),
            "train_acc": checkpoint.get("train_acc", ""),
            "train_f1": checkpoint.get("train_f1", ""),
            "train_auc": checkpoint.get("train_auc", ""),
            "val_loss": checkpoint.get("val_loss", ""),
            "val_acc": checkpoint.get("val_acc", ""),
            "val_f1": checkpoint.get("val_f1", ""),
            "val_auc": checkpoint.get("val_auc", ""),
        }

    else:
        modelo.load_state_dict(checkpoint)
        epoch = "desconocida"
        checkpoint_info = {"epoch": epoch}

    return modelo, checkpoint_info


def guardar_csv(csv_path, data_dir, modelo_path, checkpoint_info, loss, acc, f1, auc, matriz):

    tn, fp, fn, tp = matriz[0, 0], matriz[0, 1], matriz[1, 0], matriz[1, 1]
    epoch = checkpoint_info.get("epoch", "desconocida")

    with open(csv_path, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "data_dir",
            "modelo_path",
            "epoch",
            "checkpoint_train_loss",
            "checkpoint_train_acc",
            "checkpoint_train_f1",
            "checkpoint_train_auc",
            "checkpoint_val_loss",
            "checkpoint_val_acc",
            "checkpoint_val_f1",
            "checkpoint_val_auc",
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
            checkpoint_info.get("train_loss", ""),
            checkpoint_info.get("train_acc", ""),
            checkpoint_info.get("train_f1", ""),
            checkpoint_info.get("train_auc", ""),
            checkpoint_info.get("val_loss", ""),
            checkpoint_info.get("val_acc", ""),
            checkpoint_info.get("val_f1", ""),
            checkpoint_info.get("val_auc", ""),
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
    csv_path, matriz_path = preparar_rutas_resultados(args)

    print("device:", device)

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    else:
        print("cuda no disponible")

    dataset = SurFakeDataset(args.data_dir)

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    modelo, checkpoint_info = cargar_modelo(args.modelo_path)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_f1, test_auc, matriz = evaluar(
        modelo,
        test_loader,
        criterion,
        nombre="test"
    )

    guardar_csv(
        csv_path,
        args.data_dir,
        args.modelo_path,
        checkpoint_info,
        test_loss,
        test_acc,
        test_f1,
        test_auc,
        matriz
    )

    guardar_matriz_img(
        matriz_path,
        matriz,
        "Matriz de confusion - data_completa2/test"
    )

    print("\ntest final")
    print("checkpoint:", args.modelo_path)
    print("epoch checkpoint:", checkpoint_info.get("epoch", "desconocida"))
    print("checkpoint train loss:", checkpoint_info.get("train_loss", ""))
    print("checkpoint train acc :", checkpoint_info.get("train_acc", ""))
    print("checkpoint train f1  :", checkpoint_info.get("train_f1", ""))
    print("checkpoint train auc :", checkpoint_info.get("train_auc", ""))
    print("checkpoint val loss  :", checkpoint_info.get("val_loss", ""))
    print("checkpoint val acc   :", checkpoint_info.get("val_acc", ""))
    print("checkpoint val f1    :", checkpoint_info.get("val_f1", ""))
    print("checkpoint val auc   :", checkpoint_info.get("val_auc", ""))
    print(f"test loss: {test_loss:.4f}")
    print(f"test acc : {test_acc:.4f}")
    print(f"test f1  : {test_f1:.4f}")
    print(f"test auc : {test_auc:.4f}")
    print("\nmatriz confusion")
    print("filas: true real=0, true fake=1")
    print("cols : pred real=0, pred fake=1")
    print(matriz)
    print("csv:", csv_path)
    print("matriz imagen:", matriz_path)


if __name__ == "__main__":
    main()
