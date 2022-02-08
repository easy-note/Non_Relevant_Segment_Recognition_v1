
import os
import glob
import natsort
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby

import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

import datetime
from csv import DictWriter

import imageio


def encode_list(s_list): # run-length encoding from list
    return [[len(list(group)), key] for idx, (key, group) in enumerate(groupby(s_list))] # [[length, value], [length, value]...]

def cumsum(assets_df):
    class_list = assets_df['class'].values.tolist()
    encode_data = pd.DataFrame(data=encode_list(class_list), columns=['length', 'class']) # [length, value]
        
    # print('\nencode_data')
    # print(encode_data)

    # arrange data
    runlength_df = pd.DataFrame(range(0,0)) # empty df
    runlength_df = runlength_df.append(encode_data)

    # print('\nrunlength_df')
    # print(runlength_df)

    # Nan -> 0, convert to int
    runlength_df = runlength_df.fillna(0).astype(int)

    # split data, class // both should be same length
    runlength_class = runlength_df['class'] # class info
    runlength_model = runlength_df['length'] # run length info of model prediction

    # data processing for barchart
    data = np.array(runlength_model.to_numpy()) # width
    data_cum = data.cumsum(axis=0) # for calc start index, 누적 합

    starts_list = []
    for i, frame_class in enumerate(runlength_class) :
        widths = data[i]
        starts_list.append(data_cum[i] - widths)

    runlength_df['st_pos'] = starts_list
    runlength_df['ed_pos'] = data_cum

    # print('\runlength_df')
    # print(runlength_df)

    return runlength_df


def gen_gif(assets_df, target_fps, save_path):
    import numpy as np

    os.makedirs(save_path, exist_ok=True)

    frame_list = assets_df.frame_path.values.tolist()
    ssim_score = assets_df.ssim_score.values.tolist()

    runlength_df = cumsum(assets_df)
    
    rs_target_df = runlength_df[runlength_df['class']==1]
    nrs_target_df = runlength_df[runlength_df['class']==3]

    speed_sec = {'duration':0.5}

    # for nrs
    for st_pos, ed_pos in zip(nrs_target_df['st_pos'], nrs_target_df['ed_pos']):
        target_frame = frame_list[st_pos:ed_pos+1]

        target_ssim_score = ssim_score[st_pos:ed_pos]

        mean = round(np.mean(target_ssim_score), 4)
        std = round(np.std(target_ssim_score), 4)
        max_x = round(max(target_ssim_score), 4)
        min_n = round(min(target_ssim_score), 4)
        
        images = []
        for frame in target_frame:
            images.append(imageio.imread(frame))

        if ed_pos - st_pos >= 60 * target_fps: # 60초 이상 유지될 때
            gif_save_path = os.path.join(save_path, str(target_fps)+'_fps', 'NRS', 'LONG')
        else:
            gif_save_path = os.path.join(save_path, str(target_fps)+'_fps', 'NRS', 'SHORT')


        os.makedirs(gif_save_path, exist_ok=True)
        gif_name = '{}-{}_({}, {}, {}, {}).gif'.format(st_pos, ed_pos, min_n, mean, max_x, std)
        imageio.mimsave(os.path.join(gif_save_path, gif_name), images, **speed_sec)


    # for rs
    for st_pos, ed_pos in zip(rs_target_df['st_pos'], rs_target_df['ed_pos']):
        target_frame = frame_list[st_pos:ed_pos+1]

        target_ssim_score = ssim_score[st_pos:ed_pos]
        
        mean = round(np.mean(target_ssim_score), 4)
        std = round(np.std(target_ssim_score), 4)
        max_x = round(max(target_ssim_score), 4)
        min_n = round(min(target_ssim_score), 4)
        
        images = []
        for frame in target_frame:
            images.append(imageio.imread(frame))

        if ed_pos - st_pos >= 60 * target_fps: # 60초 이상 유지될 때
            gif_save_path = os.path.join(save_path, str(target_fps)+'_fps', 'RS', 'LONG')
        else:
            gif_save_path = os.path.join(save_path, str(target_fps)+'_fps', 'RS', 'SHORT')


        os.makedirs(gif_save_path, exist_ok=True)
        gif_name = '{}-{}_({}, {}, {}, {}).gif'.format(st_pos, ed_pos, min_n, mean, max_x, std)
        imageio.mimsave(os.path.join(gif_save_path, gif_name), images, **speed_sec)


'''
def cumsum(self, data):
        pos_info = []
        for i in range(data.shape[1]):
            _data = data[:, i]
            cum_val = _data[0]
            st_pos = 0
            ed_pos = 0
            X = []
            for d in _data[1:]:
                if d == cum_val:
                    ed_pos += 1
                else:
                    X.append([st_pos, ed_pos, cum_val])
                    cum_val = d
                    st_pos = ed_pos
                    ed_pos += 1
            if ed_pos < len(data):
                X.append([st_pos, len(data), cum_val])
            pos_info.append(X)
        return pos_info
'''

if __name__ == '__main__':
    with open('./assets/L_1/L_1-1_fps.pickle', 'rb') as f:
        data1 = pickle.load(f)

    with open('./assets/L_1/L_1-5_fps.pickle', 'rb') as f:
        data2 = pickle.load(f)

    gen_gif(assets_df=data2, target_fps=5, save_path='./test/04_GS4_99_L_1/gif') # ssim_result/gangbuksamsung_127case/04_GS4_99_L_1)
    # cumsum(data2)
   