import os
import glob

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from PIL import Image
from natsort import natsorted

from core.config.data_info import data_transforms, data_transforms2, theator_data_transforms

class DBDataset(Dataset): 

    def __init__(self, args, DB_path): 
        self.args = args
        self.img_list = glob.glob(os.path.join(DB_path, '*.jpg')) # ALL img into DB path
        self.img_list = natsorted(self.img_list) # sorting

        if self.args.experiment_type == 'ours':
            if self.args.model == 'mobile_vit':
                d_transforms = data_transforms2
            else:    
                d_transforms = data_transforms
        elif self.args.experiment_type == 'theator':
            d_transforms = theator_data_transforms

        self.aug = d_transforms['test']
        # self.aug = data_transforms2['test']
        # self.aug = data_transforms['test']
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        # parsing DB img idx
        video_name, db_idx = os.path.splitext(os.path.basename(img_path))[0].split('-') # ~/01_G_01_R_999_ch1_1-0000000001.jpg

        return {'img': img,
                'db_idx': db_idx,
                'img_path': img_path}

class IntervalSampler(Sampler):
    
    def __init__(self, data_source, interval):
        self.data_source = data_source
        self.interval = interval
    
    def __iter__(self):
        return iter(range(0, len(self.data_source), self.interval))
    
    def __len__(self):
        return len(self.data_source)

        