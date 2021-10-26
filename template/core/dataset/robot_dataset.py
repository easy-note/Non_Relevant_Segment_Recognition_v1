import os
import sys
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
import pandas as pd
import natsort

from torch.utils.data import Dataset
from core.config.data_info import data_transforms, theator_data_transforms

from core.config.patients_info import train_videos, val_videos, hem_train_videos, hem_val_videos, hem_test_videos


class RobotDataset(Dataset):
    def __init__(self, args, state) :
        super().__init__()

        self.args = args
        self.mode_hem = False
        self.ids = None

        self.IB_ratio = self.args.IB_ratio
        self.random_seed = self.args.random_seed

        self.img_list = [] # img
        self.label_list = [] # label

        if self.args.experiment_type == 'ours':
            d_transforms = data_transforms
        elif self.args.experiment_type == 'theator':
            d_transforms = theator_data_transforms

        if args.generate_hem_mode in ['hem-bs', 'hem-vi-softmax', 'hem-vi-voting']: 
            self.mode_hem = True

        if self.mode_hem: # hem dataset 생성 O
            print('\t>>>> hem dataset 생성')
            if state == 'train':
                self.aug = d_transforms['train']
                self.patients_info_key = hem_train_videos # 60 case
            elif state == 'val':
                self.aug = d_transforms['val']
                self.patients_info_key = hem_val_videos # 20 case
            elif state == 'test':
                self.aug = d_transforms['test']
                self.patients_info_key = hem_test_videos

            print('\t>>>> Use general data for train and validation\n')
            # patients load
            self.load_patients()
            # data load
            self.load_data()

        else: # hem dataset 생성 X
            print('\n\t>>>> 이 실험에서 hem dataset 생성 따윈 없다.\n')
            if state == 'train':
                self.aug = d_transforms['train']
                self.patients_info_key = train_videos # 80 case

                # hem dataset 으로 학습
                if self.args.train_method == 'hem':
                    print('\t>>>> Use HEM data for train\n')
                    self.load_patients()
                    
                    # TODO 
                    # path 정의 >>>> dir 내부에 fold2, fold3, fold4, fold5 result
                    csv_path = os.path.join(self.args.data_base_path, 'oob_assets/HEM/MC_softmax')
                    self.load_data_from_hem_idx(csv_path)
                
                # 일반 dataset 으로 학습
                else:
                    print('\t>>>> Use general data for train\n')
                    self.load_patients()
                    self.load_data()

            elif state == 'val':
                self.aug = d_transforms['val']
                self.patients_info_key = val_videos # 20 case

                print('\t>>>> Use general data for validation\n')
                self.load_patients()
                self.load_data()

            elif state == 'test':
                self.aug = d_transforms['test']
                self.patients_info_key = val_videos


    def change_mode(self, to_hem=True): # JH 수정 : to_hem=False -> to_hem=True
        self.mode_hem = to_hem

    def set_sample_ids(self, ids):
        self.ids = ids

    def load_data(self):
        if self.args.data_version == 'v1':
            self.load_v1()
        elif self.args.data_version == 'v2':
            csv_path = os.path.join(self.args.data_base_path, 'oob_assets/V2/ROBOT')
            self.load_data_from_ver(csv_path)
        elif self.args.data_version == 'v3':
            csv_path = os.path.join(self.args.data_base_path, 'oob_assets/V3/ROBOT')
            self.load_data_from_ver(csv_path)

    def load_patients(self):
        if self.args.fold is not 'free':
            self.patients_name = self.patients_info_key[self.args.fold] # train_videos['1']
        else:
            # self.patients_name = self.args.train_videos
            ## 임의의 data list 에 대해 학습 및 테스트를 할 경우, 별도의 룰에 의거해서 지정. e.g. random으로 100개 중 80개 선택, 2 step 씩 넘어가며 선택.
            pass


    def load_v1(self):
        # TODO load dataset ver. 1 나중에 민국님과 회의 때, 논문에 사용하는지 여쭤보고 -> 필요하면 작업. 
        pass

    def load_data_from_ver(self, csv_path):

        # read oob_assets_inbody.csv, oob_assets_outofbody.csv
        read_ib_assets_df = pd.read_csv(os.path.join(csv_path, 'oob_assets_inbody.csv'), names=['img_path', 'class_idx']) # read inbody csv
        read_oob_assets_df = pd.read_csv(os.path.join(csv_path, 'oob_assets_outofbody.csv'), names=['img_path', 'class_idx']) # read inbody csv
        
        print('==> \tInbody_READ_CSV')
        print(read_ib_assets_df)
        print('\n\n')

        print('==> \tOutofbody_READ_CSV')
        print(read_oob_assets_df)
        print('\n\n')

        # select patient frame 
        print('==> \tPATIENT')
        print('|'.join(self.patients_name))
        patients_name_for_parser = [patient + '_' for patient in self.patients_name]
        print('|'.join(patients_name_for_parser))

        # select patient video
        self.ib_assets_df = read_ib_assets_df[read_ib_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]
        self.oob_assets_df = read_oob_assets_df[read_oob_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]

        # sort
        self.ib_assets_df = self.ib_assets_df.sort_values(by=['img_path'])
        self.oob_assets_df = self.oob_assets_df.sort_values(by=['img_path'])

        print('\n\n')
        print('==> \tSORT INBODY_CSV')
        print(self.ib_assets_df)
        print('\t'* 4)
        print('==> \tSORT OUTBODY_CSV')
        print(self.oob_assets_df)
        print('\n\n')

        # random_sampling and setting IB:OOB data ratio
        self.ib_assets_df = self.ib_assets_df.sample(n=int(len(self.oob_assets_df)*self.IB_ratio), replace=False, random_state=self.random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
        self.oob_assets_df = self.oob_assets_df.sample(frac=1, replace=False, random_state=self.random_seed)

        print('\n\n')
        print('==> \tRANDOM SAMPLING INBODY_CSV')
        print(self.ib_assets_df)
        print('\t'* 4)
        print('==> \tRANDOM SAMPLING OUTBODY_CSV')
        print(self.oob_assets_df)
        print('\n\n')

        # suffle 0,1
        self.assets_df = pd.concat([self.ib_assets_df, self.oob_assets_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
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

    def load_data_from_hem_idx(self, csv_path):
    
        # csv_path 내부의 모든 hem.csv (fold2, fold3, fold4, fold5) ==> 하나로 읽기
        read_hem_csv = glob(os.path.join(csv_path, '*.csv'))
        read_hem_csv = natsort.natsorted(read_hem_csv)

        hem_df_list = []
        cols = ['img_path', 'class_idx', 'HEM']

        for csv_file in read_hem_csv:
            df = pd.read_csv(csv_file, names=cols)
            hem_df_list.append(df)
    
        hem_assets_df = pd.concat(hem_df_list).reset_index(drop=True)

        # select patient frame 
        print('==> \tPATIENT')
        print('|'.join(self.patients_name))
        patients_name_for_parser = [patient + '_' for patient in self.patients_name]
        print('|'.join(patients_name_for_parser))

        # select patient video
        self.hem_assets_df = hem_assets_df[hem_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]

        # # sort
        # self.hem_hard = self.hem_assets_df[hem_assets_df['HEM']==1]
        # self.vanila = self.hem_assets_df[hem_assets_df['HEM']==0]
        # self.hem_assets_df = pd.concat([self.hem_hard, self.vanila])

        self.hem_assets_df = self.hem_assets_df.sort_values(by=['img_path'])

        print('\n\n')
        print('==> \tSORT HEM_CSV')
        print(self.hem_assets_df)

        # suffle 0,1
        self.assets_df = self.hem_assets_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        print('\n\n')
        print('==> \tFINAL SUFFLE ASSETS')
        print(self.assets_df)
        print('\n\n')

        print('\n\n')
        print('==> \tFINAL HEAD')
        print(self.assets_df.head(20))
        print('\n\n')

        # last processing
        self.img_list = self.assets_df.img_path.tolist()
        self.label_list = self.assets_df.class_idx.tolist()


    def change_labels(self, labels):
        self.label_list = labels


    def __len__(self):
        return len(self.img_list)

    # return img, label
    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.label_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        return img_path, img, label

    

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core.config.base_opts import parse_opts

    parser = parse_opts()
    args = parser.parse_args()

    a = RobotDataset(args=args, state='train') 
