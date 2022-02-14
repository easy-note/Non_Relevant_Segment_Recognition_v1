


import sys
from os import path    
base_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
print(base_path)
sys.path.append(base_path)

import os
import glob
import pandas as pd

from scripts.unit_test import test_visual_sampling


base_path = '/OOB_RECOG/logs-new'
model = 'resnet' # ['mobilenet', 'mobilevit', 'resnet']

target_path  = glob.glob(os.path.join(base_path, model+'*','hem_assets' ,'*.csv')) 

for i in target_path:
    print(i)
    
    hem_assets_path = i
    hem_df = pd.read_csv(hem_assets_path)   

    method_info = '-'.join(i.split('/')[-1].split('.')[0].split('-')[:-1])

    save_dir = os.path.join('/'.join(hem_assets_path.split('/')[:-1]), method_info)

    print(save_dir)

    test_visual_sampling.visual_flow_for_sampling(hem_df, model, save_dir)