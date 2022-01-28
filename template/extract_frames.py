import json
import os
import numpy as np
import natsort
import subprocess
import cv2
from glob import glob


data_path = '/dataset/datasets/STG_Robot/'
save_path = '/raid/img_db/ROBOT2/'


for dir_name in os.listdir(data_path): # R_0001 
    dpath = os.path.join(data_path, dir_name)
    
    file_list = glob(dpath + '/*')
    
    for input_video_path in file_list:
        file_name = input_video_path.split('/')[-1].split('.')[0]
        
        save_path2 = os.path.join(save_path, dir_name, file_name)
        os.makedirs(save_path2, exist_ok=True)
        
        cap = cv2.VideoCapture(input_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ext_frames = len(glob(save_path2 + '/*.jpg'))
        
        if frame_count != ext_frames:
            print('\nProcessing ====>\t {}\n'.format(save_path2))
            output_img_path = save_path2 + '/%010d.jpg'
            cmd = ['ffmpeg', '-i', input_video_path, '-start_number', '0', '-vsync', '0', '-vf', 'scale=512:512']
            cmd += [output_img_path]
            
            print(cmd)

            print('Running: ', " ".join(cmd))
            subprocess.run(cmd)
        