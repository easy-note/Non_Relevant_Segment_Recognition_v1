# png, gif 생성

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

def encode_list(s_list, multiply, overflow_data): # run-length encoding from list
    length = sum(1 for _ in groupby(s_list))
    return [[len(list(group))*multiply, key] if idx != length-1 else [len(list(group))*multiply-overflow_data, key] for idx, (key, group) in enumerate(groupby(s_list))] # [[length, value], [length, value]...]

def get_cnt_text(assets_df):
    rs_non_duplicate = len(assets_df[assets_df['class']==0])
    rs_duplicate = len(assets_df[assets_df['class']==1])
    nrs_non_duplicate = len(assets_df[assets_df['class']==2])
    nrs_duplicate = len(assets_df[assets_df['class']==3])

    return rs_non_duplicate, rs_duplicate, nrs_non_duplicate, nrs_duplicate


def visual_sampling(assets_df_1_fps, assets_df_5_fps, window_size, patient_no, time_th, ssim_score_th, save_path, VIDEO_FPS):

    window_size = len(assets_df_5_fps) // 5

    frame_label = assets_df_5_fps.frame_idx.values.tolist() # [0, 6, 12, 18, 24, 30, ... ]
    time_label = assets_df_5_fps.time_idx.values.tolist() # [0:00:00.00, 0:00:00.06, 0:00:00.12, 0:00:00.18, 0:00:00.24, 0:00:00.30, ... ]

    fig, ax = plt.subplots(1,1,figsize=(18,5)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

    plt.subplots_adjust(left=0.125,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0, 
            hspace=0.35)

    rs_non_duplicate, rs_duplicate, nrs_non_duplicate, nrs_duplicate = 0, 1, 2, 3

    yticks = ['class_1_fps', 'class_5_fps'] # y축 names, 순서중요

    ### for plt variable, it should be pair synk
    label_names = ['RS-non_dupliate', 'RS-dupliate', 'NRS-non_dupliate', 'NRS-dupliate']
    colors = ['moccasin', 'darkgoldenrod', 'salmon', 'red']
    alpha_ratio = [1,1,1,1]
    edge_colors = ['moccasin', 'darkgoldenrod', 'salmon', 'red']
    height = 0.5 # bar chart thic

    class_5_fps_list = assets_df_5_fps['class'].values.tolist()
    class_1_fps_list = assets_df_1_fps['class'].values.tolist()

    overflow_cnt = len(class_1_fps_list)*5 - len(class_5_fps_list)

    visual_data = {
        'class_1_fps' : class_1_fps_list,
        'class_5_fps' : class_5_fps_list,
    }

    encode_data = {}
    for y_name in yticks : # run_length coding
        
        if y_name == 'class_1_fps':
            encode_data[y_name] = pd.DataFrame(data=encode_list(visual_data[y_name], multiply=5, overflow_data=overflow_cnt), columns=[y_name, 'class']) # [length, value]

        elif y_name == 'class_5_fps':
            encode_data[y_name] = pd.DataFrame(data=encode_list(visual_data[y_name], multiply=1, overflow_data=0), columns=[y_name, 'class']) # [length, value]
        
    # print('\nencode_data')
    # print(encode_data)

    # arrange data
    runlength_df = pd.DataFrame(range(0,0)) # empty df
    for y_name in yticks :
        runlength_df = runlength_df.append(encode_data[y_name])

    # print('\nrunlength_df')
    # print(runlength_df)

    # Nan -> 0, convert to int
    runlength_df = runlength_df.fillna(0).astype(int)

    # split data, class // both should be same length
    runlength_class = runlength_df['class'] # class info
    runlength_model = runlength_df[yticks] # run length info of model prediction

    # print('\nrunlength_class')
    # print(runlength_class)

    ### draw ###
    ##### initalize label for legned, this code should be write before writing barchart #####
    init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[rs_non_duplicate], height=height, color=colors[rs_non_duplicate], alpha=alpha_ratio[rs_non_duplicate], edgecolor=edge_colors[rs_non_duplicate]) # dummy data
    init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[rs_duplicate], height=height, color=colors[rs_duplicate], alpha=alpha_ratio[rs_duplicate], edgecolor=edge_colors[rs_duplicate]) # dummy data
    init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[nrs_non_duplicate], height=height, color=colors[nrs_non_duplicate], alpha=alpha_ratio[nrs_non_duplicate], edgecolor=edge_colors[nrs_non_duplicate]) # dummy data
    init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[nrs_duplicate], height=height, color=colors[nrs_duplicate], alpha=alpha_ratio[nrs_duplicate], edgecolor=edge_colors[nrs_duplicate]) # dummy data
    # ##### #### #### #### ##### #### #### #### 

    # data processing for barchart
    data = np.array(runlength_model.to_numpy()) # width
    data_cum = data.cumsum(axis=0) # for calc start index, 누적 합

    # draw bar
    for i, frame_class in enumerate(runlength_class) :
        widths = data[i,:]
        starts= data_cum[i,:] - widths
        
        bar = ax.barh(range(len(yticks)), widths, left=starts, height=height, color=colors[frame_class], alpha=alpha_ratio[frame_class], edgecolor=edge_colors[frame_class]) # don't input label

    ### write on figure 
    # set title
    title_name = 'SSIM Score of {}'.format(patient_no)
    sub_title_name = 'NRS time threshold: {} | SSIM score threshold: {}'.format(time_th, ssim_score_th)
    fig.suptitle(title_name, fontsize=12)
    ax.set_title(sub_title_name)

    # # set xticks pre section size
    ax.set_xticks(range(0, len(frame_label), window_size)) # 9000번째 프레임 (1fps 기준으로 1프레임 = 1초 ==> 9000프레임 = 9000초) => 위치 찍어두기 용도.

    # xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]
    xtick_labels = ['{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]

    ax.set_xticklabels(xtick_labels) # xtick change
    ax.xaxis.set_tick_params(labelsize=12)
    ax.set_xlabel('Frame / Time (h:m:s:fps)', fontsize=12)

    # set yticks
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=10)	
    ax.set_ylabel('FPS', fontsize=12)

    # 8. legend
    box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

    # 9. 보조선(눈금선) 나타내기
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

    rs_non_duplicate_5fps, rs_duplicate_5fps, nrs_non_duplicate_5fps, nrs_duplicate_5fps = get_cnt_text(assets_df_5_fps)
    rs_non_duplicate_1fps, rs_duplicate_1fps, nrs_non_duplicate_1fps, nrs_duplicate_1fps = get_cnt_text(assets_df_1_fps)

    text_5_fps = ['TOTAL: {}'.format(len(assets_df_5_fps)), 'RS (non-duplicate): '+str(rs_non_duplicate_5fps), 'RS (duplicate): '+str(rs_duplicate_5fps), 'NRS (non-duplicate): '+str(nrs_non_duplicate_5fps), 'NRS (duplicate): '+str(nrs_duplicate_5fps), 'VIDEO FPS: '+str(VIDEO_FPS)]
    text_1_fps = ['TOTAL: {}'.format(len(assets_df_1_fps)), 'RS (non-duplicate): '+str(rs_non_duplicate_1fps), 'RS (duplicate): '+str(rs_duplicate_1fps), 'NRS (non-duplicate): '+str(nrs_non_duplicate_1fps), 'NRS (duplicate): '+str(nrs_duplicate_1fps), 'VIDEO FPS: '+str(VIDEO_FPS)]

    plt.text(1.0, 0.65, ' | '.join(text_5_fps))
    plt.text(1.0, 0.30, ' | '.join(text_1_fps))

    os.makedirs(save_path, exist_ok=True)

    plt.show()
    plt.savefig(os.path.join(save_path, '{}.png'.format(save_path.split('/')[-1])), dpi=500)

    plt.close(fig)



if __name__ == '__main__':
    with open('./assets/L_1/L_1-1_fps.pickle', 'rb') as f:
        data1 = pickle.load(f)

    with open('./assets/L_1/L_1-5_fps.pickle', 'rb') as f:
        data2 = pickle.load(f)

    visual_sampling(data1, data2, window_size = 5000, patient_no = 'R_1', nrs_frame_thres = '0', ssim_score_thres = '0.95', save_path=os.path.join('./ssim_result2/visual'))