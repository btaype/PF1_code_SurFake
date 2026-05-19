import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


img_exts=(".jpg",".jpeg",".png",".bmp",".webp")


class SurFakeDataset(Dataset):

    def __init__(self,root_dir):

        self.root_dir=root_dir
        self.samples=[]

        self._collect("real","real_gsd",0)
        self._collect("fake","fake_gsd",1)

        print("total muestras:",len(self.samples))


    def _collect(self,rgb_folder,gsd_folder,label):

        rgb_root=os.path.join(self.root_dir,rgb_folder)
        gsd_root=os.path.join(self.root_dir,gsd_folder)

        for dirpath,_,filenames in os.walk(rgb_root):

            for file in filenames:

                if not file.lower().endswith(img_exts):
                    continue

                rgb_path=os.path.join(dirpath,file)

                rel_path=os.path.relpath(rgb_path,rgb_root)
                rel_base=os.path.splitext(rel_path)[0]

                gsd_path=os.path.join(gsd_root,rel_base+".png")

                if os.path.exists(gsd_path):
                    self.samples.append((rgb_path,gsd_path,label))

                else:
                    print("no se encontro gsd:",gsd_path)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):

        rgb_path,gsd_path,label=self.samples[idx]

        rgb=cv2.imread(rgb_path)

        if rgb is None:
            raise FileNotFoundError(rgb_path)

        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(224,224))
        rgb=rgb.astype(np.float32)/255.0
        rgb=torch.from_numpy(rgb).permute(2,0,1)

        gsd=cv2.imread(gsd_path)

        if gsd is None:
            raise FileNotFoundError(gsd_path)

        gsd=cv2.cvtColor(gsd,cv2.COLOR_BGR2RGB)
        gsd=cv2.resize(gsd,(224,224))
        gsd=gsd.astype(np.float32)/255.0
        gsd=torch.from_numpy(gsd).permute(2,0,1)

        x=torch.cat([rgb,gsd],dim=0)

        y=torch.tensor(label,dtype=torch.long)

        return x,y