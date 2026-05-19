import csv
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from torch.utils.data import DataLoader,Subset

from ImputData import SurFakeDataset
from Modetrain import MobileNetV2_SurFake

import random
from collections import defaultdict


device="cuda" if torch.cuda.is_available() else "cpu"

data_dir="data_completa"

batch_size=12
epocas=30

lr=0.001
momentum=0.9
weight_decay=0.0001

modelo_path="modelo_actual.pth"
csv_resultados="resultados_epocas.csv"


def get_video_group(rgb_path):

    path=rgb_path.replace("\\","/")
    parts=path.split("/")

    if "real" in parts:
        idx=parts.index("real")

        return "real_"+parts[idx+1]

    if "fake" in parts:
        idx=parts.index("fake")

        tecnica=parts[idx+1]
        video_id=parts[idx+2]

        return f"fake_{tecnica}_{video_id}"

    raise ValueError(f"No se pudo identificar grupo: {rgb_path}")


def get_strata_group(rgb_path):

    path=rgb_path.replace("\\","/")
    parts=path.split("/")

    if "real" in parts:
        return "real"

    if "fake" in parts:
        idx=parts.index("fake")

        tecnica=parts[idx+1]

        return f"fake_{tecnica}"

    raise ValueError(f"No se pudo identificar clase: {rgb_path}")


def split_por_video(dataset,train_ratio=0.72,val_ratio=0.14,seed=42):

    random.seed(seed)

    grupos=defaultdict(lambda: defaultdict(list))

    for idx,(rgb_path,gsd_path,label) in enumerate(dataset.samples):

        strata=get_strata_group(rgb_path)
        video=get_video_group(rgb_path)

        grupos[strata][video].append(idx)

    train_idx=[]
    val_idx=[]
    test_idx=[]

    for strata,videos in grupos.items():

        videos_lista=list(videos.keys())

        random.shuffle(videos_lista)

        n=len(videos_lista)

        n_train=int(train_ratio*n)
        n_val=int(val_ratio*n)

        train_videos=videos_lista[:n_train]
        val_videos=videos_lista[n_train:n_train+n_val]
        test_videos=videos_lista[n_train+n_val:]

        for vg in train_videos:
            train_idx.extend(videos[vg])

        for vg in val_videos:
            val_idx.extend(videos[vg])

        for vg in test_videos:
            test_idx.extend(videos[vg])

        print(
            strata,
            "videos total:",n,
            "train:",len(train_videos),
            "val:",len(val_videos),
            "test:",len(test_videos)
        )

    random.shuffle(train_idx)

    print("imagenes train:",len(train_idx))
    print("imagenes val:",len(val_idx))
    print("imagenes test:",len(test_idx))

    return(
        Subset(dataset,train_idx),
        Subset(dataset,val_idx),
        Subset(dataset,test_idx)
    )


def metricas(y_true,y_pred,y_prob):

    acc=accuracy_score(y_true,y_pred)

    f1=f1_score(y_true,y_pred)

    try:
        auc=roc_auc_score(y_true,y_prob)

    except ValueError:
        auc=0.0

    return acc,f1,auc


def evaluar(modelo,loader,criterion,nombre="Validation"):

    modelo.eval()

    total_loss=0.0

    y_true=[]
    y_pred=[]
    y_prob=[]

    barra=tqdm(loader,total=len(loader),desc=nombre,leave=False)

    with torch.no_grad():

        for x,y in barra:

            x=x.to(device)
            y=y.to(device)

            salida=modelo(x)

            loss=criterion(salida,y)

            probs=torch.softmax(salida,dim=1)[:,1]

            preds=salida.argmax(dim=1)

            total_loss+=loss.item()*x.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            barra.set_postfix({
                "loss":f"{loss.item():.4f}",
                "gpu":torch.cuda.is_available()
            })

    avg_loss=total_loss/len(loader.dataset)

    acc,f1,auc=metricas(y_true,y_pred,y_prob)

    return avg_loss,acc,f1,auc


def train():

    print("device:",device)

    if torch.cuda.is_available():
        print("gpu:",torch.cuda.get_device_name(0))

    else:
        print("cuda no disponible")

    dataset=SurFakeDataset(data_dir)

    total=len(dataset)

    train_dataset,val_dataset,test_dataset=split_por_video(dataset)

    print("total:",total)
    print("train:",len(train_dataset))
    print("val:",len(val_dataset))
    print("test:",len(test_dataset))

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    modelo=MobileNetV2_SurFake(
        num_classes=2,
        pretrained=True
    ).to(device)

    criterion=nn.CrossEntropyLoss()

    optimizer=torch.optim.SGD(
        modelo.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    with open(csv_resultados,mode="w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss","train_acc","train_f1","train_auc",
            "val_loss","val_acc","val_f1","val_auc"
        ])

    for epoch in range(epocas):

        modelo.train()

        total_loss=0.0

        y_true=[]
        y_pred=[]
        y_prob=[]

        barra=tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch+1}/{epocas}"
        )

        for batch_idx,(x,y) in barra:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()

            salida=modelo(x)

            loss=criterion(salida,y)

            loss.backward()

            optimizer.step()

            probs=torch.softmax(salida,dim=1)[:,1]

            preds=salida.argmax(dim=1)

            total_loss+=loss.item()*x.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_prob.extend(probs.detach().cpu().numpy())

            barra.set_postfix({
                "loss":f"{loss.item():.4f}",
                "batch":f"{batch_idx+1}/{len(train_loader)}",
                "batch_size":x.size(0),
                "input":tuple(x.shape),
                "gpu":torch.cuda.is_available(),
                "vram_gb":(
                    f"{torch.cuda.memory_allocated()/1024**3:.2f}"
                    if torch.cuda.is_available()
                    else "0"
                )
            })

        train_loss=total_loss/len(train_loader.dataset)

        train_acc,train_f1,train_auc=metricas(
            y_true,y_pred,y_prob
        )

        val_loss,val_acc,val_f1,val_auc=evaluar(
            modelo,val_loader,criterion,nombre="validation"
        )

        print(
            f"\nepoch [{epoch+1}/{epocas}]\n"
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"train f1: {train_f1:.4f} | "
            f"train auc: {train_auc:.4f}\n"
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f} | "
            f"val f1: {val_f1:.4f} | "
            f"val auc: {val_auc:.4f}\n"
        )

        torch.save({

            "epoch":epoch+1,

            "model_state_dict":modelo.state_dict(),

            "optimizer_state_dict":optimizer.state_dict(),

            "train_loss":train_loss,
            "train_acc":train_acc,
            "train_f1":train_f1,
            "train_auc":train_auc,

            "val_loss":val_loss,
            "val_acc":val_acc,
            "val_f1":val_f1,
            "val_auc":val_auc,

        },modelo_path)

        with open(csv_resultados,mode="a",newline="") as f:

            writer=csv.writer(f)

            writer.writerow([
                epoch+1,

                train_loss,
                train_acc,
                train_f1,
                train_auc,

                val_loss,
                val_acc,
                val_f1,
                val_auc
            ])

    test_loss,test_acc,test_f1,test_auc=evaluar(
        modelo,test_loader,criterion,nombre="test"
    )

    print("\ntest final")

    print(f"test loss: {test_loss:.4f}")
    print(f"test acc : {test_acc:.4f}")
    print(f"test f1  : {test_f1:.4f}")
    print(f"test auc : {test_auc:.4f}")

    print("\nentrenamiento terminado")

    print("modelo guardado:",modelo_path)
    print("csv:",csv_resultados)


if __name__=="__main__":
    train()