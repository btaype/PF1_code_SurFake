import argparse
import csv
import json
import os
import torch
import torch.nn as nn

from datetime import datetime
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


def parse_args():

    parser=argparse.ArgumentParser(
        description="Entrena SurFake guardando checkpoints por epoca."
    )

    parser.add_argument("--data_dir",default=data_dir)
    parser.add_argument("--output_dir",default="experimentos")
    parser.add_argument("--nombre_experimento",default=None)
    parser.add_argument("--batch_size",type=int,default=batch_size)
    parser.add_argument("--epocas",type=int,default=epocas)
    parser.add_argument("--lr",type=float,default=lr)
    parser.add_argument("--momentum",type=float,default=momentum)
    parser.add_argument("--weight_decay",type=float,default=weight_decay)
    parser.add_argument("--porcentaje_datos",type=float,default=100.0)
    parser.add_argument("--train_count",type=int,default=720)
    parser.add_argument("--val_count",type=int,default=140)
    parser.add_argument("--seed",type=int,default=42)

    return parser.parse_args()


def crear_carpeta_experimento(args):

    if args.nombre_experimento:
        nombre=args.nombre_experimento

    else:
        fecha=datetime.now().strftime("%Y%m%d_%H%M%S")
        porcentaje=str(args.porcentaje_datos).replace(".","p")
        nombre=f"train_{fecha}_datos_{porcentaje}"

    carpeta=os.path.join(args.output_dir,nombre)
    checkpoints_dir=os.path.join(carpeta,"checkpoints")

    os.makedirs(checkpoints_dir,exist_ok=True)

    return carpeta,checkpoints_dir


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


def get_split_video_id(rgb_path):

    path=rgb_path.replace("\\","/")
    parts=path.split("/")

    if "real" in parts:
        idx=parts.index("real")
        video_id=parts[idx+1]

    elif "fake" in parts:
        idx=parts.index("fake")
        video_id=parts[idx+2]

    else:
        raise ValueError(f"No se pudo identificar video_id: {rgb_path}")

    video_id=video_id.lower()

    if video_id.startswith("video_"):
        video_id=video_id.split("_",1)[1]

    return video_id.split("_")[0]


def video_sort_key(video_id):

    return int(video_id) if video_id.isdigit() else video_id


def frame_sort_key(rgb_path):

    base=os.path.splitext(os.path.basename(rgb_path))[0]

    return int(base) if base.isdigit() else base


def seleccionar_intercalados(indices,porcentaje,dataset,seed=42):

    if porcentaje<=0 or porcentaje>100:
        raise ValueError("--porcentaje_datos debe estar entre 0 y 100")

    if porcentaje==100:
        return list(indices)

    grupos=defaultdict(list)

    for idx in indices:
        rgb_path,_,_=dataset.samples[idx]
        grupos[get_video_group(rgb_path)].append(idx)

    seleccionados=[]

    for _,items in grupos.items():

        items=sorted(
            items,
            key=lambda idx: frame_sort_key(dataset.samples[idx][0])
        )

        total=len(items)
        cantidad=max(1,round(total*porcentaje/100.0))

        if cantidad>=total:
            seleccionados.extend(items)
            continue

        if cantidad==1:
            seleccionados.append(items[total//2])
            continue

        posiciones=[
            round(i*(total-1)/(cantidad-1))
            for i in range(cantidad)
        ]

        seleccionados.extend(items[pos] for pos in posiciones)

    random.seed(seed)
    random.shuffle(seleccionados)

    return seleccionados


def reducir_subset_por_video(subset,porcentaje,seed=42):

    if not isinstance(subset,Subset):
        raise TypeError("Se esperaba un torch.utils.data.Subset")

    indices=seleccionar_intercalados(
        subset.indices,
        porcentaje,
        subset.dataset,
        seed=seed
    )

    return Subset(subset.dataset,indices)


def get_strata_group(rgb_path):

    path=rgb_path.replace("\\","/")
    parts=path.split("/")

    if "real" in parts:

        return "real"

    if "fake" in parts:

        idx=parts.index("fake")

        tecnica=parts[idx+1]

        return f"fake_{tecnica}"

    raise ValueError(f"no se pudo identificar la clase: {rgb_path}")


def split_por_video(dataset,train_count=720,val_count=140,seed=42):

    random.seed(seed)

    grupos=defaultdict(lambda: defaultdict(list))
    split_video_ids=set()
    video_to_split_id={}

    for idx,(rgb_path,gsd_path,label) in enumerate(dataset.samples):

        strata=get_strata_group(rgb_path)
        video=get_video_group(rgb_path)
        split_video_id=get_split_video_id(rgb_path)

        grupos[strata][video].append(idx)
        split_video_ids.add(split_video_id)
        video_to_split_id[video]=split_video_id

    split_video_ids=sorted(split_video_ids,key=video_sort_key)

    train_video_ids=set(split_video_ids[:train_count])
    val_video_ids=set(split_video_ids[train_count:train_count+val_count])

    train_idx=[]
    val_idx=[]
    test_idx=[]

    for strata,videos in grupos.items():

        videos_lista=list(videos.keys())

        videos_lista.sort(key=lambda video:(video_sort_key(video_to_split_id[video]),video))

        n=len(videos_lista)

        train_videos=[
            video for video in videos_lista
            if video_to_split_id[video] in train_video_ids
        ]
        val_videos=[
            video for video in videos_lista
            if video_to_split_id[video] in val_video_ids
        ]
        test_videos=[
            video for video in videos_lista
            if (
                video_to_split_id[video] not in train_video_ids
                and video_to_split_id[video] not in val_video_ids
            )
        ]

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

    args=parse_args()
    carpeta_experimento,checkpoints_dir=crear_carpeta_experimento(args)
    csv_epocas=os.path.join(carpeta_experimento,"resultados_epocas.csv")
    csv_test=os.path.join(carpeta_experimento,"resultados_test.csv")
    config_path=os.path.join(carpeta_experimento,"config.json")

    print("device:",device)

    if torch.cuda.is_available():
        print("gpu:",torch.cuda.get_device_name(0))

    else:
        print("cuda no disponible")

    print("experimento:",carpeta_experimento)
    print("porcentaje datos train:",args.porcentaje_datos)

    with open(config_path,mode="w",encoding="utf-8") as f:
        json.dump(vars(args),f,indent=2)

    dataset=SurFakeDataset(args.data_dir)

    total=len(dataset)

    train_dataset,val_dataset,test_dataset=split_por_video(
        dataset,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed
    )

    train_original=len(train_dataset)

    train_dataset=reducir_subset_por_video(
        train_dataset,
        args.porcentaje_datos,
        seed=args.seed
    )

    print("total:",total)
    print("train original:",train_original)
    print("train usado:",len(train_dataset))
    print("val:",len(val_dataset))
    print("test:",len(test_dataset))

    train_loader=DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    modelo=MobileNetV2_SurFake(
        num_classes=2,
        pretrained=True
    ).to(device)

    criterion=nn.CrossEntropyLoss()

    optimizer=torch.optim.SGD(
        modelo.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    with open(csv_epocas,mode="w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss","train_acc","train_f1","train_auc",
            "val_loss","val_acc","val_f1","val_auc"
        ])

    mejor_val_f1=-1.0

    for epoch in range(args.epocas):

        modelo.train()

        total_loss=0.0

        y_true=[]
        y_pred=[]
        y_prob=[]

        barra=tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch+1}/{args.epocas}"
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
            f"\nepoch [{epoch+1}/{args.epocas}]\n"
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"train f1: {train_f1:.4f} | "
            f"train auc: {train_auc:.4f}\n"
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f} | "
            f"val f1: {val_f1:.4f} | "
            f"val auc: {val_auc:.4f}\n"
        )

        checkpoint={

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

            "config":vars(args),

        }

        checkpoint_path=os.path.join(
            checkpoints_dir,
            f"epoch_{epoch+1:03d}.pth"
        )

        ultimo_path=os.path.join(carpeta_experimento,"ultimo.pth")

        torch.save(checkpoint,checkpoint_path)
        torch.save(checkpoint,ultimo_path)

        if val_f1>mejor_val_f1:
            mejor_val_f1=val_f1
            mejor_path=os.path.join(carpeta_experimento,"mejor_val_f1.pth")
            torch.save(checkpoint,mejor_path)

        with open(csv_epocas,mode="a",newline="") as f:

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

    with open(csv_test,mode="w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow([
            "test_loss","test_acc","test_f1","test_auc",
            "checkpoint_final","checkpoint_mejor_val_f1"
        ])

        writer.writerow([
            test_loss,
            test_acc,
            test_f1,
            test_auc,
            os.path.join(carpeta_experimento,"ultimo.pth"),
            os.path.join(carpeta_experimento,"mejor_val_f1.pth")
        ])

    print("\nentrenamiento terminado")

    print("carpeta experimento:",carpeta_experimento)
    print("checkpoints por epoca:",checkpoints_dir)
    print("modelo final:",os.path.join(carpeta_experimento,"ultimo.pth"))
    print("mejor val f1:",os.path.join(carpeta_experimento,"mejor_val_f1.pth"))
    print("csv epocas:",csv_epocas)
    print("csv test:",csv_test)


if __name__=="__main__":
    train()
