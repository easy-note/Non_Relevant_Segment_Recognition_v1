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
from core.config.patients_info import train_videos, val_videos
from core.utils.heuristic_sampling import HeuristicSampler


class RobotDataset(Dataset):
    def __init__(self, args, state) :
        super().__init__()

        self.args = args
        self.mode_hem = False
        self.ids = None

        self.IB_ratio = self.args.IB_ratio
        self.random_seed = self.args.random_seed
        self.state = state

        self.img_list = [] # img
        self.label_list = [] # label

        ### load dataset from under setting ###
        self.patients_name = [] # init
        assets_type = '' # ['sub', 'meta', 'hem'] 
        ### ### ###

        self.wise_sampling_mode = self.args.use_wise_sample

        if self.args.experiment_type == 'ours':
            d_transforms = data_transforms
        elif self.args.experiment_type == 'theator':
            d_transforms = theator_data_transforms
        
        if self.args.mini_fold is not 'general': # hem dataset 생성 (60/20) // offline 사용
            if state == 'train':
                self.aug = d_transforms['train']
                
                # split to 60/20 from 80 case
                self.patients_name = self.set_patient_per_mini_fold(train_videos[self.args.fold], mode='train') # args.fold's train dataset(80 case) to 60 case
                assets_type='sub'

                # self.wise_sampling_mode = True
                
            elif state == 'val':
                self.aug = d_transforms['val']

                # split to 60/20 from 80 case
                self.patients_name = self.set_patient_per_mini_fold(train_videos[self.args.fold], mode='val') # args.fold's train dataset(80 case) to 20 case
                assets_type='meta'

                self.wise_sampling_mode = False
                
            elif state == 'test':
                pass

        elif self.args.mini_fold is 'general': # hem dataset 생성 X (80/20) // online, offline 모두 사용
            if state == 'train':   
                self.aug = d_transforms['train']     
                
                # 20 case / 80 case
                self.patients_name = train_videos[self.args.fold]

                if 'offline' in self.args.hem_extract_mode and self.args.stage is 'hem_train': 
                    assets_type='hem'

                else: # general_train, bs-emb1-online, bs-emb2-online, bs-emb3-online
                    # TODO - general train and online 
                    assets_type='sub'

                    # self.wise_sampling_mode = True

                    
            elif state == 'val':
                self.aug = d_transforms['val']
                
                # 20 case
                self.patients_name = val_videos[self.args.fold]
                assets_type='sub'

                self.wise_sampling_mode = False

            elif state == 'test':
                pass

        self.load_data(assets_type)

    def set_patient_per_mini_fold(self, patients_list, mode='train'):
        patients_list = natsort.natsorted(patients_list)

        patients_dict = {
            '0': patients_list[:20],
            '1': patients_list[20:40],
            '2': patients_list[40:60],   
            '3': patients_list[60:80]
        }

        if mode == 'train':
            return list(set(patients_list)-set(patients_dict[self.args.mini_fold]))
        elif mode == 'val':
            return patients_dict[self.args.mini_fold]

    def load_data(self, assets_type):
        support_type = ['sub', 'meta', 'hem']
        assert assets_type in support_type, 'NOT SOPPORT TYPE'

        assets_root_path = ''

        if self.args.data_version == 'v1':
            assert False, 'Not Yet Support'
        elif self.args.data_version == 'v2':
            assets_root_path = os.path.join(self.args.data_base_path, 'oob_assets/V2/ROBOT')
        elif self.args.data_version == 'v3':
            assets_root_path = os.path.join(self.args.data_base_path, 'oob_assets/V3/ROBOT')

        ## set meta/sub assets path
        assets_path = {
            'meta_ib':os.path.join(assets_root_path, 'oob_assets_inbody-fps=5.csv'),
            'meta_oob':os.path.join(assets_root_path, 'oob_assets_outofbody-fps=5.csv'),
            'sub_ib':os.path.join(assets_root_path, 'oob_assets_inbody.csv'),
            'sub_oob':os.path.join(assets_root_path, 'oob_assets_outofbody.csv'),
        }
        
        if assets_type == 'sub': # 1fps
            print('[LOAD FROM SUBSET]')
            self.load_data_from_assets(assets_path['sub_ib'], assets_path['sub_oob'])
        
        elif assets_type == 'meta': # 30fps -> 5fps
            print('[LOAD FROM METASET]')
            self.load_data_from_assets(assets_path['meta_ib'], assets_path['meta_oob'])
        
        elif assets_type == 'hem': # from extracted csv
            print('[LOAD FROM HEMSET]')
            self.load_data_from_hem_assets()

    def load_v1(self):
        # TODO load dataset ver. 1 나중에 민국님과 회의 때, 논문에 사용하는지 여쭤보고 -> 필요하면 작업. 
        pass

    def load_data_from_assets(self, ib_assets_csv_path, oob_assets_csv_path):

        print('IB_ASSETS_PATH: {}'.format(ib_assets_csv_path))
        print('OOB_ASSETS_PATH: {}'.format(oob_assets_csv_path))

        read_ib_assets_df = pd.read_csv(ib_assets_csv_path, names=['img_path', 'class_idx']) # read inbody csv
        read_oob_assets_df = pd.read_csv(oob_assets_csv_path, names=['img_path', 'class_idx']) # read inbody csv

        '''
        # read oob_assets_inbody.csv, oob_assets_outofbody.csv
        read_ib_assets_df = pd.read_csv(os.path.join(csv_path, 'oob_assets_inbody.csv'), names=['img_path', 'class_idx']) # read inbody csv
        read_oob_assets_df = pd.read_csv(os.path.join(csv_path, 'oob_assets_outofbody.csv'), names=['img_path', 'class_idx']) # read inbody csv
        '''
        
        print('==> \tInbody_READ_CSV')
        print(read_ib_assets_df)
        print('\n\n')

        print('==> \tOutofbody_READ_CSV')
        print(read_oob_assets_df)
        print('\n\n')

        # select patient frame 
        print('==> \tPATIENT ({})'.format(len(self.patients_name)))
        print('|'.join(self.patients_name))
        patients_name_for_parser = [patient + '_' for patient in self.patients_name]
        print('|'.join(patients_name_for_parser))

        # select patient video
        ib_assets_df = read_ib_assets_df[read_ib_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]
        oob_assets_df = read_oob_assets_df[read_oob_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]

        # sort
        ib_assets_df = ib_assets_df.sort_values(by=['img_path'])
        oob_assets_df = oob_assets_df.sort_values(by=['img_path'])

        if self.wise_sampling_mode:
            # hueristic_sampling
            assets_df = pd.concat([ib_assets_df, oob_assets_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            assets_df = assets_df.sort_values(by='img_path')

            hueristic_sampler = HeuristicSampler(assets_df, self.args)
            assets_df = hueristic_sampler.final_assets

            assets_df['HEM'] = [0]*len(assets_df)

            os.makedirs(os.path.join(self.args.save_path, 'HEM_ASSETS'), exist_ok=True)
            assets_df.to_csv(os.path.join(self.args.save_path, 'HEM_ASSETS','STAGE={}-wise_sampling.csv'.format(self.args.stage)))

        else:
            print('\n\n')
            print('==> \tSORT INBODY_CSV')
            print(ib_assets_df)
            print('\t'* 4)
            print('==> \tSORT OUTBODY_CSV')
            print(oob_assets_df)
            print('\n\n')

            # random_sampling and setting IB:OOB data ratio
            # HG 21.11.08 error fix, ratio로 구성 불가능 할 경우 전체 set 모두 사용
            max_ib_count, target_ib_count = len(ib_assets_df), int(len(oob_assets_df)*self.IB_ratio)
            sampling_ib_count = max_ib_count if max_ib_count < target_ib_count else target_ib_count
            print('Random sampling from {} to {}'.format(max_ib_count, sampling_ib_count))

            ib_assets_df = ib_assets_df.sample(n=sampling_ib_count, replace=False, random_state=self.random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
            oob_assets_df = oob_assets_df.sample(frac=1, replace=False, random_state=self.random_seed)

            print('\n\n')
            print('==> \tRANDOM SAMPLING INBODY_CSV')
            print(ib_assets_df)
            print('\t'* 4)
            print('==> \tRANDOM SAMPLING OUTBODY_CSV')
            print(oob_assets_df)
            print('\n\n')

            # suffle 0,1
            assets_df = pd.concat([ib_assets_df, oob_assets_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            print('\n\n')
            print('==> \tFINAL ASSETS ({})'.format(len(assets_df)))
            print(assets_df)
            print('\n\n')

            print('\n\n')
            print('==> \tFINAL HEAD')
            print(assets_df.head(20))
            print('\n\n')


        # last processing
        self.img_list = assets_df.img_path.tolist()
        self.label_list = assets_df.class_idx.tolist()

    def load_data_from_hem_assets(self):
        
        # self.args.restore_path 에서 version0, 1, 2, 3 에 대한 hem.csv 읽고
        self.restore_path = '/'.join(self.args.restore_path.split('/')[:-1])
        
        # csv_path 내부의 모든 hem.csv (fold2, fold3, fold4, fold5) ==> 하나로 합침
        read_hem_csv = glob(os.path.join(self.restore_path, '*', '*-*-*.csv'))

        read_hem_csv = natsort.natsorted(read_hem_csv)
        
        print(read_hem_csv)

        hem_df_list = []
        cols = ['img_path', 'class_idx', 'HEM']

        for csv_file in read_hem_csv:
            df = pd.read_csv(csv_file, names=cols)
            hem_df_list.append(df)
    
        hem_assets_df = pd.concat(hem_df_list, ignore_index=True).reset_index(drop=True)

        hem_assets_df.to_csv(os.path.join(self.args.restore_path, 'hem_assets.csv'))

        # select patient frame 
        print('==> \tPATIENT ({})'.format(len(self.patients_name)))
        print('|'.join(self.patients_name))
        patients_name_for_parser = [patient + '_' for patient in self.patients_name]
        print('|'.join(patients_name_for_parser))

        # select patient video
        hem_assets_df = hem_assets_df[hem_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]

        # # sort
        # self.hem_hard = self.hem_assets_df[hem_assets_df['HEM']==1]
        # self.vanila = self.hem_assets_df[hem_assets_df['HEM']==0]
        # self.hem_assets_df = pd.concat([self.hem_hard, self.vanila])

        hem_assets_df = hem_assets_df.sort_values(by=['img_path'])

        print('\n\n')
        print('==> \tSORT HEM_CSV')
        print(hem_assets_df)

        # suffle 0,1
        assets_df = hem_assets_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        print('\n\n')
        print('==> \tFINAL SUFFLE ASSETS ({})'.format(len(assets_df)))
        print(assets_df)
        print('\n\n')

        print('\n\n')
        print('==> \tFINAL HEAD')
        print(assets_df.head(20))
        print('\n\n')

        # last processing
        self.img_list = assets_df.img_path.tolist()
        self.label_list = assets_df.class_idx.tolist()


    def number_of_rs_nrs(self):
        return self.label_list.count(0) ,self.label_list.count(1)
        

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
