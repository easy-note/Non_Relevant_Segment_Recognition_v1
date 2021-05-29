import os
import glob2
import glob
import cv2
from PIL import Image
import random
import numpy as np
import pandas as pd
from pandas import DataFrame as df

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

IB_CLASS, OOB_CLASS = (0,1)


def make_oob_csv(data_dir, save_dir) : 
    
    IB_dir_name, OOB_dir_name = ['InBody', 'OutBody']

    class_dir_name = ['InBody', 'OutBody']

    print(data_dir)
    print(save_dir)
    
    img_path_list = []
    class_list = []

    # calc path, class
    for class_path in class_dir_name :
        print('processing... ', os.path.join(data_dir, class_path))

        # init
        temp_path = []
        temp_class = []

        temp_path = glob.glob(os.path.join(data_dir, class_path, '*.jpg'))

        if class_path == IB_dir_name :
            temp_class = [IB_CLASS]*len(temp_path)

        if class_path == OOB_dir_name :
            temp_class = [OOB_CLASS]*len(temp_path)

        img_path_list += temp_path
        class_list += temp_class

    # save 
    save_df = df({
        'img_path' : img_path_list,
        'class_idx' : class_list })    # 모든 이미지 path와 class 정보 저장


    save_df.to_csv(os.path.join(save_dir, 'oob_assets_path.csv'), mode='w', index=False)



class CAMIO_Dataset(Dataset):
    def __init__(self, csv_path, patient_name, is_train, random_seed, IB_ratio):
        self.is_train = is_train
        self.csv_path = csv_path 
        self.aug = data_transforms['train'] if is_train else data_transforms['val']
        
        self.img_list = [] # img
        self.label_list = [] # label

        read_assets_df = pd.read_csv(csv_path) # read csv
        
        print('\n\n')
        print('==> \tPATIENT')
        print('|'.join(patient_name))
        patient_name_for_parser = [patient + '_' for patient in patient_name]
        print('|'.join(patient_name_for_parser))
        
        print('==> \tREAD_CSV')
        print(read_assets_df)
        print('\n\n')


        # select patient video
        self.assets_df = read_assets_df[read_assets_df['img_path'].str.contains('|'.join(patient_name_for_parser))]

        # seperate df by class
        self.ib_df = self.assets_df[self.assets_df['class_idx']==IB_CLASS]
        self.oob_df = self.assets_df[self.assets_df['class_idx']==OOB_CLASS]

        print('\n\n')
        print('==> \tINBODY_CSV')
        print(self.ib_df)
        print('\t'* 4)
        print('==> \tOUTBODY_CSV')
        print(self.oob_df)
        print('\n\n')

        # sort
        self.ib_df = self.ib_df.sort_values(by=['img_path'])
        self.oob_df = self.oob_df.sort_values(by=['img_path'])

        print('\n\n')
        print('==> \tSORT INBODY_CSV')
        print(self.ib_df)
        print('\t'* 4)
        print('==> \tSORT OUTBODY_CSV')
        print(self.oob_df)
        print('\n\n')

        # random_sampling and setting IB:OOB data ratio
        self.ib_df = self.ib_df.sample(n=len(self.oob_df)*IB_ratio, replace=False, random_state=random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
        self.oob_df = self.oob_df.sample(frac=1, replace=False, random_state=random_seed)

        print('\n\n')
        print('==> \tRANDOM SAMPLING INBODY_CSV')
        print(self.ib_df)
        print('\t'* 4)
        print('==> \tRANDOM SAMPLING OUTBODY_CSV')
        print(self.oob_df)
        print('\n\n')

        # suffle 0,1
        self.assets_df = pd.concat([self.ib_df, self.oob_df]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        print('\n\n')
        print('==> \tFINAL ASSETS')
        print(self.assets_df)
        print('\n\n')

        print('\n\n')
        print('==> \tFINAL HEAD')
        print(self.assets_df.head(20))
        print('\n\n')

        # last processing
        self.img_list = self.assets_df.img_path.tolist()
        self.label_list = self.assets_df.class_idx.tolist()
        
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.label_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        return img, label

