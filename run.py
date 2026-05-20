import os
import cv2
import torch
import numpy as np

from types import SimpleNamespace
from models.uprightnet_model import UprightNet


base_dir="data_completa"

fake_input=os.path.join(base_dir,"fake")
real_input=os.path.join(base_dir,"real")

fake_output=os.path.join(base_dir,"fake_gsd")
real_output=os.path.join(base_dir,"real_gsd")

batch_size=8

img_exts=(".jpg",".jpeg",".png",".bmp",".webp")


def build_opt():

    opt=SimpleNamespace()

    opt.gpu_ids=[0] if torch.cuda.is_available() else []

    opt.isTrain=False
    opt.mode="ResNet"
    opt.dataset="interiornet"
    opt.checkpoints_dir="./checkpoints"
    opt.name="test_local"

    opt.lr=0.0004
    opt.lr_policy="step"
    opt.lr_decay_epoch=10
    opt.epoch_count=1
    opt.niter=1
    opt.niter_decay=1
    opt.backprop_eig=1

    opt.w_pose=0.0
    opt.w_cam=0.0
    opt.w_up=0.0
    opt.w_grad=0.0

    return opt


def cargar_img(path):

    img=cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"No se pudo abrir: {path}")

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=cv2.resize(img,(384,288))

    img=img.astype(np.float32)/255.0

    img=torch.from_numpy(img).permute(2,0,1)

    return img


def juntar_imgs(input_root,output_root):

    items=[]

    for dirpath,_,filenames in os.walk(input_root):

        for filename in filenames:

            if not filename.lower().endswith(img_exts):
                continue

            img_path=os.path.join(dirpath,filename)

            rel_path=os.path.relpath(img_path,input_root)

            rel_base=os.path.splitext(rel_path)[0]

            out_path=os.path.join(output_root,rel_base+".npy")

            if os.path.exists(out_path):
                continue

            items.append((img_path,out_path))

    return items


def guardar_gsd_batch(gsd_batch,out_paths):

    gsd_batch=gsd_batch.detach().cpu()

    for gsd,out_path in zip(gsd_batch,out_paths):

        x=gsd.permute(1,2,0).numpy()

        x=cv2.resize(x,(224,224))

        x=(x+1.0)/2.0
        x=np.clip(x,0.0,1.0)

        x=(x*255).astype(np.uint8)

        x=cv2.cvtColor(x,cv2.COLOR_RGB2BGR)

        out_path=os.path.splitext(out_path)[0]+".png"

        os.makedirs(os.path.dirname(out_path),exist_ok=True)

        cv2.imwrite(out_path,x)


def procesar_items(modelo,items,nombre):

    total=len(items)

    print(f"\nProcesando {nombre}: {total} imagenes pendientes")

    if total==0:
        print(f"No hay imagenes nuevas en {nombre}")
        return

    modelo.switch_to_eval()

    procesadas=0
    fallidas=0

    for start in range(0,total,batch_size):

        batch_items=items[start:start+batch_size]

        imgs=[]
        out_paths=[]

        for img_path,out_path in batch_items:

            try:
                img=cargar_img(img_path)

                imgs.append(img)

                out_paths.append(out_path)

            except Exception as e:
                fallidas+=1
                print(f"Error leyendo {img_path}: {e}")

        if len(imgs)==0:
            continue

        batch=torch.stack(imgs,dim=0)

        try:

            with torch.no_grad():

                modelo.input=batch

                modelo.forward()

                gsd_batch=modelo.pred_up_geo_unit

                guardar_gsd_batch(gsd_batch,out_paths)

            procesadas+=len(imgs)

        except RuntimeError as e:

            print(f"\nError CUDA {start}: {e}")

            raise e

        if procesadas%200==0 or procesadas==total:
            print(f"{nombre}: {procesadas}/{total} procesadas | fallidas: {fallidas}")

    print(f"\nFinalizado {nombre}")
    print(f"Procesadas: {procesadas}")
    print(f"Fallidas: {fallidas}")


def main():

    print("cuda disponible:",torch.cuda.is_available())

    opt=build_opt()

    modelo=UprightNet(opt,_isTrain=False)

    fake_items=juntar_imgs(fake_input,fake_output)

    real_items=juntar_imgs(real_input,real_output)

    procesar_items(modelo,fake_items,"fake")

    procesar_items(modelo,real_items,"real")

    print("\nlisto")

    print("gsd fake guardado en:",fake_output)

    print("gsd real guardado en:",real_output)


if __name__=="__main__":
    main()