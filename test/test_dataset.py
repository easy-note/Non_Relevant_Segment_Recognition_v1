import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from natsort import natsorted

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

IB_CLASS, OOB_CLASS = (0,1)

class OOB_DB_Dataset(Dataset): 

    def __init__(self, DB_path): 
        self.img_list = glob.glob(os.path.join(DB_path, '*.jpg')) # ALL img into DB path
        self.img_list = natsorted(self.img_list) # sorting

        self.aug = data_transforms['test']
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        # parsing DB img idx
        video_name, idx = os.path.splitext(os.path.basename(img_path))[0].split('-') # ~/01_G_01_R_999_ch1_1-0000000001.jpg
        

        return {'img': img,
                'DB_idx': idx,
                'img_path': img_path}


# IT'S NOT UNIFORM SAMPLER
# test => [0,5,10,15...] step size
# gradcam => [0, 9, 11, 13 ...]
# https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
# Dataloader(OOB_DB_Dataset, batch_sampler=[[0,1,2], [3,4,5], [6,7], [8,9]])
class IDX_Sampler():
    def __init__(self, idx_list, batch_size):
        self.idx_list = idx_list
        self.batch_size = batch_size
    
    def __iter__(self):
        batches = []

        start_pos = 0
        end_pos = len(self.idx_list)

        for _ in range(start_pos, end_pos + self.batch_size, self.batch_size):
            batch_idx = self.idx_list[start_pos : start_pos + self.batch_size]

            if batch_idx != []:
                batches.append(batch_idx)
            
            start_pos = start_pos + self.batch_size
        
        return iter(batches)