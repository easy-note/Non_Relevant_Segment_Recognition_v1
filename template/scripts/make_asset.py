import sys
from os import path    

base_path = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(base_path)
sys.path.append(base_path+'/core/accessory/RepVGG')


import os
from glob import glob
import natsort
import numpy as np
import torch
import csv
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.config.base_opts import parse_opts
from core.api.trainer import CAMIO
from core.config.data_info import data_transforms, data_transforms2
from core.utils.misc import get_inference_model_path


def save_list_to_csv(save_filepath, list, mode):
    with open(save_filepath, mode) as f:
        writer = csv.writer(f)
        for row in list:
            writer.writerow(row)

class inferset(torch.utils.data.Dataset):
    def __init__(self, data_path, bsz):
        self.batch_size = bsz
        self.data_path = data_path
        
        self.X = []
        fps = 30
        
        for dir_name in natsort.natsorted(os.listdir(data_path)): # R_0001 
            dpath = os.path.join(data_path, dir_name)
            
            _video_list = natsort.natsorted(os.listdir(dpath))
            video_list = []
            
            if 'ch' in _video_list[0]:
                for vpath in _video_list:
                    if 'ch1' in vpath:
                        video_list.append(vpath)
            else:
                v_len = len(_video_list)
                if v_len > 1:
                    video_list = _video_list[:v_len//2]
                else:
                    video_list = _video_list
                    
            for dir_name2 in video_list:
                dpath2 = os.path.join(dpath, dir_name2)
                
                print('Current Processed : {}'.format(dpath2))
                
                file_list = glob(dpath2 + '/*.jpg')
                file_list = natsort.natsorted(file_list)[::fps]
                
                self.X += file_list
                
    def set_aug(self, aug):
        self.aug = aug
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        fpath = self.X[index]
        
        img = Image.open(fpath)
        img = self.aug(img)#.unsqueeze(0)
        
        return fpath, img


data_path = '/raid/img_db/ROBOT2/'
save_path = '/raid/img_db/oob_assets/V3/ROBOT/'
fps = 30

inferset = inferset(data_path, 256)

# model_list = ['mobilenet', 'resnet', 'mvit']
model_list = ['mobilenet']

for model in model_list:
    if model == 'mvit':
        aug = data_transforms2['val']
    else:
        aug = data_transforms['val']

    common_path = '/code/OOB_Recog/logs-{}'.format(model)

    model_path_list = [
        common_path + '/{}-rs-general-trial:1-fold:1/TB_log/version_0'.format(model),
        common_path + '/{}-ws-general-trial:1-fold:1/TB_log/version_0'.format(model),
        common_path + '/{}-rs-proxy-type2-2-step-trial:1-fold:1/TB_log/version_0'.format(model),
        common_path + '/{}-ws-proxy-type2-2-step-trial:1-fold:1/TB_log/version_0'.format(model),
        common_path + '/{}-rs-proxy-type2-4-step-trial:1-fold:1/TB_log/version_0'.format(model),
        common_path + '/{}-ws-proxy-type2-4-step-trial:1-fold:1/TB_log/version_0'.format(model),
    ]

    method_list = [
        '{}-rs-general'.format(model), 
        '{}-ws-general'.format(model), 
        '{}-rs-proxy-base'.format(model), 
        '{}-ws-proxy-base'.format(model), 
        '{}-rs-proxy-wrong'.format(model),
        '{}-ws-proxy-wrong'.format(model),
    ]

    inferset.set_aug(aug)
    

    for method, _model_path in zip(method_list, model_path_list):
        if 'proxy' in method:
            stage = 'hem_train'
            hem_mode = 'hem-emb-online'
            
            if 'base' in method:
                utype = 2
            else:
                utype = 4
        else:
            stage = 'general_train'
            hem_mode = ''
            utype = 2
            
        model_path = get_inference_model_path(_model_path)

        parser = parse_opts()
        args = parser.parse_args()
        args.pretrained = True
        args.data_base_path = '/raid/img_db'
        args.num_gpus = 1
        args.use_lightning_style_save = True 
        args.restore_path = model_path
        args.hem_extract_mode = hem_mode
        args.stage = stage
        args.update_type2 = True
        args.update_type = utype
        

        cam_io = CAMIO(args)
        model = cam_io.model.cuda().eval()

        inbody_list = []
        outbody_list = []
        
        d_loader = DataLoader(inferset, batch_size=256, shuffle=False, num_workers=6)

        for path_list, imgs in tqdm(d_loader, desc='Inference ... '):
            y_hat = model(imgs.cuda())
            y_hat = torch.nn.functional.softmax(y_hat, -1)
            output = torch.argmax(y_hat, -1)
            
            for fpath, out in zip(path_list, output):
                if out.item() == 1:
                    outbody_list.append([fpath, 1])
                else:
                    inbody_list.append([fpath, 0])
                        
        # inbody_list = natsort.natsorted(inbody_list)
        # outbody_list = natsort.natsorted(outbody_list)
        
        save_list_to_csv(os.path.join(save_path, 'oob_assets_outofbody-soft-label-{}.csv'.format(method)), outbody_list, 'w')
        save_list_to_csv(os.path.join(save_path, 'oob_assets_inbody-soft-label-{}.csv'.format(method)), inbody_list, 'w')