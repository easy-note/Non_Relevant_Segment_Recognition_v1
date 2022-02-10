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
from core.config.data_info import data_transforms, data_transforms2, theator_data_transforms

from core.config.patients_info import train_videos as robot_train_videos
from core.config.patients_info import val_videos as robot_val_videos
from core.config.patients_info import unsup_videos as robot_unsup_videos

from core.config.patients_info_lapa import train_videos as lapa_train_videos
from core.config.patients_info_lapa import val_videos as lapa_val_videos

from core.utils.heuristic_sampling import HeuristicSampler
from core.config.assets_info import oob_assets_save_path

from scripts.unit_test.test_visual_sampling import visual_flow_for_sampling

from core.utils.misc import parse_patient_for_lapa

class RobotDataset_new(Dataset):

    def __init__(self, args, state, dataset_type='ROBOT', minifold='0', wise_sample=False, all_sample=False, use_metaset=False, appointment_assets_path='') : # 애초에 appointment_assets_path 에서 불러올꺼면 객체생성부터 초기화해주어야 함.
        super().__init__()


        self.args = args
        self.state = state
        self.minifold = minifold
        self.dataset_type = dataset_type # ['ROBOT', 'LAPA']   

        ## set self var from args
        self.model = self.args.model
        self.IB_ratio = self.args.IB_ratio
        self.random_seed = self.args.random_seed
        self.WS_ratio = self.args.WS_ratio

        self.experiment_type = self.args.experiment_type
        
        self.fold = self.args.fold  

        ## set load setting
        ''' TIP
        # dataset = ['ROBOT', 'LAPA']
        # wise sample : wise_sample[True], all_sample[Fasle]
        # all sample : wise_sampple[Fasle], all_sample[True]
        # random sample : wise_sample[Fasle], all_sample[False]
        # subset : use_metaset[Fasle]
        # metaset : use_metaset[True]
        # appoinemnet assets path 설정되어 되어 있으면 wise sample, all sample 모두 무시 / use_metaset 무시
        '''
        self.wise_sampling_mode = wise_sample
        self.all_sampling_mode = all_sample
        self.use_metaset = use_metaset
        self.appointment_assets_path = appointment_assets_path

        ## get patinets video
        if self.dataset_type == 'ROBOT':
            train_videos, val_videos = robot_train_videos, robot_val_videos
        elif self.dataset_type == 'LAPA':
            train_videos, val_videos = lapa_train_videos, lapa_val_videos
        
        ### ### ###
        ## init self var
        self.img_list = [] # img
        self.label_list = [] # label
        self.assets_df = None # 사용한 patinet 정보들 파싱처리
        self.patients_name = [] # init

        if self.experiment_type == 'ours':
            if self.model == 'mobile_vit':
                d_transforms = data_transforms2
            else:    
                d_transforms = data_transforms
        elif self.experiment_type == 'theator':
            d_transforms = theator_data_transforms

        # augmentation setting, patient setting, according to state(train, val, test)
        if state == 'train': # 80 case
            self.aug = d_transforms['train']     
            self.patients_name = train_videos[self.fold]
                
        elif state == 'val': # 20 case
            self.aug = d_transforms['val']
            self.patients_name = val_videos[self.fold]
            self.wise_sampling_mode = False # 혹시 실수할까봐 어차피 validation set 은 wise sampling 하면안됨.
        
        elif state == 'train_mini': # 60 case
            if self.dataset_type == 'ROBOT':
                minifold_to_patients = {
                    '0': [], # nothing,
                    '1': train_videos[self.fold][20:],
                    '2': train_videos[self.fold][:20] + train_videos[self.fold][40:],
                    '3': train_videos[self.fold][:40] + train_videos[self.fold][60:],   
                    '4': train_videos[self.fold][:60],
                }
            elif self.dataset_type == 'LAPA':
                if self.fold == '1':
                    minifold_to_patients = {
                        '0': [], # nothing,
                        '1': train_videos[self.fold][50:],
                        '2': train_videos[self.fold][:50] + train_videos[self.fold][100:],
                        '3': train_videos[self.fold][:100] + train_videos[self.fold][150:],   
                        '4': train_videos[self.fold][:150],
                    }
                elif self.fold == '2':
                    minifold_to_patients = {
                        '0': [], # nothing,
                        '1': train_videos[self.fold][40:],
                        '2': train_videos[self.fold][:40] + train_videos[self.fold][80:],
                        '3': train_videos[self.fold][:80] + train_videos[self.fold][120:],   
                        '4': train_videos[self.fold][:120],
                    }
            
            self.aug = d_transforms['train']     
            self.patients_name = minifold_to_patients[self.minifold]

        elif state == 'val_mini': # 20 case
            if self.dataset_type == 'ROBOT':
                minifold_to_patients = {
                    '0': [], # nothing,
                    '1': train_videos[self.fold][:20],
                    '2': train_videos[self.fold][20:40],
                    '3': train_videos[self.fold][40:60],   
                    '4': train_videos[self.fold][60:],
                }
            elif self.dataset_type == 'LAPA':
                if self.fold == '1':
                    minifold_to_patients = {
                        '0': [], # nothing,
                        '1': train_videos[self.fold][:50],
                        '2': train_videos[self.fold][50:100],
                        '3': train_videos[self.fold][100:150],
                        '4': train_videos[self.fold][150:200],
                    }

                elif self.fold == '2':
                    minifold_to_patients = {
                        '0': [], # nothing,
                        '1': train_videos[self.fold][:40],
                        '2': train_videos[self.fold][40:80],
                        '3': train_videos[self.fold][80:120],
                        '4': train_videos[self.fold][120:],
                    }

            self.aug = d_transforms['val']
            self.patients_name = minifold_to_patients[self.minifold]
            self.wise_sampling_mode = False # 혹시 실수할까봐 어차피 validation set 은 wise sampling 하면안됨.
        
        elif state == 'test':
            pass
        
        self.load_data(self.appointment_assets_path)
    
    def load_data(self, appointment_assets_path):

        if appointment_assets_path is not '':
            print('[LOAD FROM APPOINTMENT]')
            print('====> {}'.format(appointment_assets_path))
            self.load_data_from_appointment_assets() # load from speicific path
        
        else: # assets type check
            ## set meta/sub assets path
            if self.dataset_type == 'ROBOT':       
                assets_root_path = os.path.join(oob_assets_save_path['oob_assets_v3_robot_save_path'])

                assets_path = {
                    'meta_ib':os.path.join(assets_root_path, 'oob_assets_inbody-fps=5.csv'),
                    'meta_oob':os.path.join(assets_root_path, 'oob_assets_outofbody-fps=5.csv'),
                    'sub_ib':os.path.join(assets_root_path, 'oob_assets_inbody.csv'),
                    'sub_oob':os.path.join(assets_root_path, 'oob_assets_outofbody.csv'),
                    }

            elif self.dataset_type == 'LAPA':
                assets_root_path = os.path.join(oob_assets_save_path['vihub_assets_v3_lapa_save_path'])

                assets_path = {
                    'meta_ib':os.path.join(assets_root_path, 'PP-VIHUB_ALL-assets_rs-fps=5-update.csv'),
                    'meta_oob':os.path.join(assets_root_path, 'PP-VIHUB_ALL-assets_nrs-fps=5-update.csv'),
                    'sub_ib':os.path.join(assets_root_path, 'PP-VIHUB_ALL-assets_rs-fps=1-update.csv'),
                    'sub_oob':os.path.join(assets_root_path, 'PP-VIHUB_ALL-assets_nrs-fps=1-update.csv'),              
                    }


            if self.use_metaset: # 30fps -> 5fps
                
                print('[LOAD FROM METASET]')
                self.load_data_from_assets(assets_path['meta_ib'], assets_path['meta_oob'])

            else:
                if self.args.experiment_sub_type == 'semi':
                    if self.args.model == 'resnet18':
                        model = 'resnet18'
                    elif self.args.model == 'mobilenetv3_large_100':
                        model = 'mobilenet'
                    elif self.args.model == 'mobile_vit':
                        model = 'mvit'
                    
                    unsup_asset_ib_path = os.path.join(assets_root_path, 'oob_assets_inbody-soft-label-{}-{}.csv'.format(model, self.args.semi_data))
                    unsup_asset_oob_path = os.path.join(assets_root_path, 'oob_assets_outofbody-soft-label-{}-{}.csv'.format(model, self.args.semi_data))
                    print('[LOAD FROM SEMI-SUPERVISED]')
                    self.load_data_from_semi_assets(assets_path['sub_ib'], assets_path['sub_oob'], unsup_asset_ib_path, unsup_asset_oob_path)
                    
                else:
                    print('[LOAD FROM SUBSET]')
                    self.load_data_from_assets(assets_path['sub_ib'], assets_path['sub_oob'])
                
    
    def load_data_from_semi_assets(self, ib_assets_csv_path, oob_assets_csv_path, ib_assets_csv_path2, oob_assets_csv_path2):
        read_ib_assets_df = pd.read_csv(ib_assets_csv_path, names=['img_path', 'class_idx']) # read inbody csv
        read_oob_assets_df = pd.read_csv(oob_assets_csv_path, names=['img_path', 'class_idx']) # read inbody csv
        
        read_ib_assets_df2 = pd.read_csv(ib_assets_csv_path2, names=['img_path', 'class_idx']) # read inbody csv
        read_oob_assets_df2 = pd.read_csv(oob_assets_csv_path2, names=['img_path', 'class_idx']) # read inbody csv

        # select patient frame 
        self.patients_name = robot_train_videos[self.args.fold]
        self.patients_name2 = robot_unsup_videos
        
        
        patients_name_for_parser = [patient + '_' for patient in self.patients_name]
        patients_name_for_parser2 = [patient + '_' for patient in self.patients_name2]

        # select patient video
        ib_assets_df = read_ib_assets_df[read_ib_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]
        oob_assets_df = read_oob_assets_df[read_oob_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]
        
        ib_assets_df2 = read_ib_assets_df2[read_ib_assets_df2['img_path'].str.contains('|'.join(patients_name_for_parser2))]
        oob_assets_df2 = read_oob_assets_df2[read_oob_assets_df2['img_path'].str.contains('|'.join(patients_name_for_parser2))]

        # sort
        ib_assets_df = ib_assets_df.sort_values(by=['img_path'])
        oob_assets_df = oob_assets_df.sort_values(by=['img_path'])
        ib_assets_df2 = ib_assets_df2.sort_values(by=['img_path'])
        oob_assets_df2 = oob_assets_df2.sort_values(by=['img_path'])
        
        if self.wise_sampling_mode:
            assets_df = pd.concat([ib_assets_df, oob_assets_df])
            assets_df2 = pd.concat([ib_assets_df2, oob_assets_df2])
        
            # hueristic_sampling
            assets_df = assets_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True).sort_values(by='img_path')
            assets_df2 = assets_df2.sample(frac=1, random_state=self.random_seed).reset_index(drop=True).sort_values(by='img_path')

            hueristic_sampler = HeuristicSampler(assets_df, self.args)
            hueristic_sampler2 = HeuristicSampler(assets_df2, self.args)
            assets_df = hueristic_sampler.final_assets
            assets_df2 = hueristic_sampler2.final_assets
            
            self.split_patient = len(assets_df)
            assets_df = pd.concat([assets_df, assets_df2])
            

            if self.state == 'train': # save plt only in trainset
                assets_df['HEM'] = [0]*len(assets_df)

                assets_df_save_dir = os.path.join(self.args.save_path, 'train_assets', '{}set_stage-{}'.format(self.state, self.args.train_stage))
                os.makedirs(assets_df_save_dir, exist_ok=True)

                assets_df.to_csv(os.path.join(assets_df_save_dir, 'stage={}-wise_sampling.csv'.format(self.args.train_stage)))

                try: # 혹시, error날 경우 pass (plt warining 가능)
                    pass
                    # visual_flow_for_sampling(assets_df, self.args.model, assets_df_save_dir, window_size=9000, section_num=2) # sampling visalization
                except:
                    pass


        else:            
            # HG 21.11.30 all sampling mode = True 라면 IB ratio적용 x => 모두 사용
            if not self.all_sampling_mode: # default = False
                # random_sampling and setting IB:OOB data ratio
                # HG 21.11.08 error fix, ratio로 구성 불가능 할 경우 전체 set 모두 사용
                max_ib_count, target_ib_count = len(ib_assets_df), int(len(oob_assets_df)*self.IB_ratio)
                sampling_ib_count = max_ib_count if max_ib_count < target_ib_count else target_ib_count

                ib_assets_df = ib_assets_df.sample(n=sampling_ib_count, replace=False, random_state=self.random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
                oob_assets_df = oob_assets_df.sample(frac=1, replace=False, random_state=self.random_seed)
                
                max_ib_count, target_ib_count = len(ib_assets_df2), int(len(oob_assets_df2)*self.IB_ratio)
                sampling_ib_count = max_ib_count if max_ib_count < target_ib_count else target_ib_count

                ib_assets_df2 = ib_assets_df2.sample(n=sampling_ib_count, replace=False, random_state=self.random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
                oob_assets_df2 = oob_assets_df2.sample(frac=1, replace=False, random_state=self.random_seed)

            # suffle 0,1
            assets_df = pd.concat([ib_assets_df, oob_assets_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            assets_df2 = pd.concat([ib_assets_df2, oob_assets_df2]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
           
            self.split_patient = len(assets_df)
            assets_df = pd.concat([assets_df, assets_df2])
            

            if self.state == 'train': # save plt only in trainset
                assets_df['HEM'] = [0]*len(assets_df)

                assets_df_save_dir = os.path.join(self.args.save_path, 'train_assets', '{}set_stage-{}'.format(self.state, self.args.train_stage))
                os.makedirs(assets_df_save_dir, exist_ok=True)

                assets_df.to_csv(os.path.join(assets_df_save_dir, 'stage={}-random_sampling.csv'.format(self.args.train_stage)))

                try: # 혹시, error날 경우 pass (plt warining 가능)
                    pass
                    # visual_flow_for_sampling(assets_df, self.args.model, assets_df_save_dir, window_size=9000, section_num=2) # sampling visalization
                except:
                    pass

        # last processing
        self.img_list = assets_df.img_path.tolist()
        self.label_list = assets_df.class_idx.tolist()

        self.assets_df = assets_df
        
        

    def load_data_from_assets(self, ib_assets_csv_path, oob_assets_csv_path): # check self.all_sampling_mode, check self.wise_sampling_mode

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
        # print(read_ib_assets_df)
        print('\n\n')

        print('==> \tOutofbody_READ_CSV')
        # print(read_oob_assets_df)
        print('\n\n')

        # select patient frame 
        print('==> \tPATIENT ({})'.format(len(self.patients_name)))
        print('|'.join(self.patients_name))
        patients_name_for_parser = [patient + '_' for patient in self.patients_name]
        # print('|'.join(patients_name_for_parser))

        # select patient video
        ib_assets_df = read_ib_assets_df[read_ib_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]
        oob_assets_df = read_oob_assets_df[read_oob_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]

        # sort
        ib_assets_df = ib_assets_df.sort_values(by=['img_path'])
        oob_assets_df = oob_assets_df.sort_values(by=['img_path'])

        # for test code
        '''
        ib_assets_df = ib_assets_df.sample(n=300, replace=False, random_state=self.random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
        oob_assets_df = oob_assets_df.sample(n=300, replace=False, random_state=self.random_seed)
        '''
        
        

        # hueristic_sampling
        if self.wise_sampling_mode:
            print('\n\n\t ==> HUERISTIC SAMPLING ... IB_RATIO: {}, WS_RATIO: {}\n\n'.format(self.IB_ratio, self.WS_ratio))
            assets_df = pd.concat([ib_assets_df, oob_assets_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            assets_df = assets_df.sort_values(by='img_path')

            hueristic_sampler = HeuristicSampler(assets_df, self.args)
            assets_df = hueristic_sampler.final_assets

        else: # all sampling or random sampling
            # sampling mode = True 라면 IB ratio적용 x => 모두 사용
            if not self.all_sampling_mode: # default = False
                print('\n\n\t ==> RANDOM SAMPLING ... IB_RATIO: {}\n\n'.format(self.IB_ratio))
                # random_sampling and setting IB:OOB data ratio
                # ratio로 구성 불가능 할 경우 전체 set 모두 사용
                max_ib_count, target_ib_count = len(ib_assets_df), int(len(oob_assets_df)*self.IB_ratio)
                sampling_ib_count = max_ib_count if max_ib_count < target_ib_count else target_ib_count
                print('Random sampling from {} to {}'.format(max_ib_count, sampling_ib_count))

                ib_assets_df = ib_assets_df.sample(n=sampling_ib_count, replace=False, random_state=self.random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
                oob_assets_df = oob_assets_df.sample(frac=1, replace=False, random_state=self.random_seed)    
    
            # suffle 0,1
            assets_df = pd.concat([ib_assets_df, oob_assets_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            print('\n\n')
            print('==> \tFINAL ASSETS ({})'.format(len(assets_df)))
            # print(assets_df)
            print('\n\n')


            '''
            if self.state == 'train': # save plt only in trainset
                assets_df['HEM'] = [0]*len(assets_df)

                assets_df_save_dir = os.path.join(self.args.save_path, 'train_assets', '{}set_stage-{}'.format(self.state, self.args.stage))
                os.makedirs(assets_df_save_dir, exist_ok=True)

                assets_df.to_csv(os.path.join(assets_df_save_dir, 'stage={}-wise_sampling.csv'.format(self.args.stage)))

                try: # 혹시, error날 경우 pass (plt warining 가능)
                    pass
                    # visual_flow_for_sampling(assets_df, self.args.model, assets_df_save_dir, window_size=9000, section_num=2) # sampling visalization
                except:
                    pass
            '''

        # last processing
        self.img_list = assets_df.img_path.tolist()
        self.label_list = assets_df.class_idx.tolist()

        self.assets_df = assets_df

    def load_data_from_appointment_assets(self): # when you want to load from specific csv path (i.e, train hem model / train stage baby model)
        
        print('\n\n\t ==> APPOINTMENT ASSETS PATH: {}'.format(self.appointment_assets_path))

        appointment_assets_df = pd.read_csv(self.appointment_assets_path) # read appointment csv

        # select patient frame 
        print('==> \tPATIENT ({})'.format(len(self.patients_name)))
        print('|'.join(self.patients_name))
        patients_name_for_parser = [patient + '_' for patient in self.patients_name]
        # print('|'.join(patients_name_for_parser))

        # select patient video
        assets_df = appointment_assets_df[appointment_assets_df['img_path'].str.contains('|'.join(patients_name_for_parser))]

        # sort & shuffle
        assets_df = assets_df.sort_values(by=['img_path'])
        assets_df = assets_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        print('\n==== FIANL STAGE ASSTES ====')
        print(assets_df)
        print('==== FIANL STAGE ASSTES ====\n')
        
        # last processing
        self.img_list = assets_df.img_path.tolist()
        self.label_list = assets_df.class_idx.tolist()

        self.assets_df = assets_df

    def number_of_rs_nrs(self):
        return self.label_list.count(0) ,self.label_list.count(1)

    # 해당 robot_dataset의 patinets별 assets 개수 (hem train할때 valset만들어서 hem_helper의 args.hem_per_patinets에서 사용됨.)     
    def number_of_patient_rs_nrs(self):
        patient_per_dic = {}

        val_assets_df = self.assets_df

        if self.dataset_type == 'ROBOT':
            val_assets_df['patient'] = val_assets_df.img_path.str.split('/').str[4]
        
        elif self.dataset_type == 'LAPA':
            val_assets_df['patient'] = val_assets_df['img_path'].apply(parse_patient_for_lapa) # hospital - vihub

        total_rs_count = len(val_assets_df[val_assets_df['class_idx']==0])
        total_nrs_count = len(val_assets_df[val_assets_df['class_idx']==1])

        patients_list = list(set(val_assets_df['patient']))
        patients_list = natsort.natsorted(patients_list)

        for patient in patients_list:
            patient_df = val_assets_df[val_assets_df['patient']==patient]
            # print(patient_df)

            patient_rs_count = len(patient_df[patient_df['class_idx']==0])
            patient_nrs_count = len(patient_df[patient_df['class_idx']==1])

            # print(patient_rs_count, patient_nrs_count)

            patient_per_dic.update(
                {
                    patient : {
                    'rs': patient_rs_count,
                    'nrs': patient_nrs_count,
                    'rs_ratio': patient_rs_count/total_rs_count,
                    'nrs_ratio': patient_nrs_count/total_nrs_count
                    } 
                }
            )

        return patient_per_dic

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
