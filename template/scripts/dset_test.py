import sys
from os import path    
base_path = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(base_path)
sys.path.append(base_path+'/core/accessory/RepVGG')

import os
import pandas as pd
import pytorch_lightning as pl
from core.api.trainer import CAMIO
from core.config.base_opts import parse_opts
from core.config.patients_info import train_videos, val_videos, unsup_videos

# parser = parse_opts()
# args = parser.parse_args()
# args.pretrained = True
# args.data_base_path = '/raid/img_db'
# args.experiment_sub_type = 'unsup-general'
# args.stage = 'general_train'
# args.mini_fold = 'general'
# args.num_gpus = 1
# args.data_version == 'v3'
# args.use_lightning_style_save = True


# x = CAMIO(args)

# x.setup(stage='fit')

# import numpy as np

# lbs = np.array(x.trainset.label_list)
# print(sum( lbs == 1 ))
# print(sum( lbs == 0 ))


base_path = '/raid/img_db/oob_assets/V3/ROBOT'


train_patients_name = train_videos['1']
val_patients_name = val_videos['1']

train_patients = [patient + '_' for patient in train_patients_name]
val_patients = [patient + '_' for patient in val_patients_name]


# meta
# sub_in = pd.read_csv(base_path + '/oob_assets_inbody.csv', names=['img_path', 'class_idx'])
# sub_out = pd.read_csv(base_path + '/oob_assets_outofbody.csv', names=['img_path', 'class_idx'])

# sub_in_train = sub_in[sub_in['img_path'].str.contains('|'.join(train_patients))]
# sub_in_val = sub_in[sub_in['img_path'].str.contains('|'.join(val_patients_name))]

# sub_out_train = sub_out[sub_out['img_path'].str.contains('|'.join(train_patients))]
# sub_out_val = sub_out[sub_out['img_path'].str.contains('|'.join(val_patients_name))]


# # sub
# meta_in = pd.read_csv(base_path + '/oob_assets_inbody-fps=5.csv', names=['img_path', 'class_idx'])
# meta_out = pd.read_csv(base_path + '/oob_assets_outofbody-fps=5.csv', names=['img_path', 'class_idx'])


# meta_in_train = meta_in[meta_in['img_path'].str.contains('|'.join(train_patients))]
# meta_in_val = meta_in[meta_in['img_path'].str.contains('|'.join(val_patients_name))]

# meta_out_train = meta_out[meta_out['img_path'].str.contains('|'.join(train_patients))]
# meta_out_val = meta_out[meta_out['img_path'].str.contains('|'.join(val_patients_name))]


# print(len(meta_in_train), len(meta_out_train))
# print(len(meta_in_val), len(meta_out_val))

# print()

# print(len(sub_in_train), len(sub_out_train))
# print(len(sub_in_val), len(sub_out_val))



import json
from glob import glob
import natsort

base_path = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V3/TBE'
file_list = natsort.natsorted(glob(base_path + '/*json'))

patient_dict = {}

for fpath in file_list:
    patient_no = 'R_' + fpath.split('/')[-1].split('_')[4] + '_'
    
    if patient_no in patient_dict:
        patient_dict[patient_no].append(fpath)
    else:
        patient_dict[patient_no] = [fpath]
    
event_cnt = {
    'train': 0,
    'val': 0,
}
        
# train_patients = [patient + '_' for patient in train_patients_name]
# val_patients = [patient + '_' for patient in val_patients_name]
        
for key, f_list in patient_dict.items():
    state = 'train'
    
    if key in val_patients:
        state = 'val'
    
    for fi in range(len(f_list)):
        with open(file_list[fi], 'r') as f:
            data = json.load(f)

        frames = data['totalFrame']
        annos = data['annotations']
        
        l_anno = len(annos)
        
        for idx, anno in enumerate(annos):
            if idx+1 == l_anno:
                if anno['end'] >= frames-1 and fi+1 != len(f_list):
                    with open(file_list[fi+1], 'r') as f:
                        t_data = json.load(f)
                        
                    t_annos = t_data['annotations']
                    
                    if t_annos[0]['start'] >= 0:
                        event_cnt[state] += 1
            else:
                event_cnt[state] += 1
                
    
print(event_cnt)