"""
Create final form of 'info_dict' for model test.

Check list
    1. Is the annotation frame longer than the video frame?
    2. Are frames all integers?
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import glob
import matplotlib
import time
import json

from pandas import DataFrame as df

import torch
import pytorch_lightning as pl

matplotlib.use('Agg')


# check over frame and modify last annotation info
def check_anno_over_frame(anno_info:list, video_len): # over frame 존재하지 않을경우 = True, over frame 존재할 경우 = False  
    has_not_over_frame = False
    
    last_start, last_end = anno_info[-1]
    
    if last_end < video_len : 
        has_not_over_frame = True

    # modify
    if not(has_not_over_frame) :
        anno_info[-1] = [last_start, video_len-1]
        print('\t\t\t *** ANNO LAST FRAME END : {} | VIDEO_LEN : {}'.format(last_end, video_len))

    return has_not_over_frame, anno_info

def check_anno_int(anno_info:list): # anno int = True, anno Float = False
    is_int = True
    
    for start, end in anno_info : 
        if (not isinstance(start, int)) and (not isinstance(end, int)) :
            is_int = False
            break
    
    # modify
    if not(is_int) :
        for i, (start, end) in enumerate(anno_info) :
            anno_info[i] = [int(math.floor(start)), int(math.floor(end))]
            print('\t\t\t *** ANNO FLOAT FRAME : [{}, {}] | REFINED FRAME : {}'.format(start, end, anno_info[i]))

    return is_int, anno_info


def gettering_information_for_oob(video_root_path, anno_root_path, inference_assets_root_path, video_set, mode) : # paring video from annotation info
    """
    Generate 'info_dict' (Incomplete form of 'info_dict') 

    Args:
        video_root_path: Video root path.
        anno_root_path: Annotation file root path.
        inference_assets_root_path: Tensor form test dataset root path.
        video_set: Video set for test.
        mode: For Robot or Lapa? Default - Robot.

    Returns:
        info_dict: 
            info_dict = {
                'video': [],
                'anno': [],
                'inference_assets' : []
            }
    """

    print('\n\n\n\t\t\t ### STARTING DEF [gettering_information_for_robot] ### \n\n')

    info_dict = {
        'video': [],
        'anno': [],
        'inference_assets' : []
    }

    all_video_path = []
    all_anno_path = []

    if mode == 'ROBOT' : 
        fps = 30
        
        video_ext_list = ['mp4']
        for ext in video_ext_list :
            all_video_path.extend(glob.glob(video_root_path +'/*.{}'.format(ext)))
        
        all_anno_path = glob.glob(anno_root_path + '/*.csv') # all annotation file list
    
    elif mode == 'LAPA' :
        fps = 60
        video_ext_list = ['mp4', 'MP4', 'mpg']
        
        for ext in video_ext_list :
            all_video_path.extend(glob.glob(video_root_path +'/*.{}'.format(ext)))
        
        all_anno_path = glob.glob(anno_root_path + '/*.json') # all annotation file list

        ##### except video file ######
        all_video_path.remove(os.path.join(video_root_path, '01_G_01_L_423_xx0_01.MP4'))
        all_anno_path.remove(os.path.join(anno_root_path, '01_G_01_L_423_xx0_01_OOB_16.json'))

    else :
        assert False, 'ONLY SUPPORT MODE [ROBOT, LAPA] | Input mode : {}'.format(mode)

    # dpath = os.path.join(video_root_path) # video_root path

    print('NUMBER OF TOTAL VIDEO FILE : ', len(all_video_path))
    print('NUMBER OF TOTAL ANNOTATION FILE : ', len(all_anno_path))
    print('')

    all_video_path_df = df(all_video_path, columns=['video_path'])
    all_anno_path_df = df(all_anno_path, columns=['anno_path'])

    print(all_video_path_df)
    print(all_anno_path_df)

    for video_no in video_set : # get target video
        # video_path_list = sorted([vfile for vfile in all_video_path if os.path.basename(vfile).startswith(video_no)])
        # anno_path_list = sorted([anno_file for anno_file in all_anno_path if os.path.basename(anno_file).startswith(video_no)])

        # find video_no in video_file
        video_path_df = all_video_path_df[all_video_path_df['video_path'].str.contains(video_no + '_')]

        # sort video_path_df for sync for video slice
        video_path_df = video_path_df.sort_values(by=['video_path'], axis=0, ascending=True)

        # init video and annotation paring path info
        pair_info = df(range(0,0), columns=['video_path', 'anno_path'])

        # video & annotation pairing
        for i in range(len(video_path_df)) : 
            video_file_name = os.path.splitext(os.path.basename(video_path_df['video_path'].iloc[i]))[0] # video_name with out extension
            anno_path_series = all_anno_path_df[all_anno_path_df['anno_path'].str.contains(video_file_name+'_OOB')]['anno_path'] # find annotation file based in video_file_name
            video_path_series = video_path_df.iloc[i]

            info = {
                'video_path':list(video_path_series)[0],
                'anno_path':np.nan if len(list(anno_path_series))==0 else list(anno_path_series)[0]
            }

            pair_info=pair_info.append(info, ignore_index=True)
        
        pair_info = pair_info.fillna('') # fill na -> ""
        
        print(pair_info)

        # df -> list 
        video_path_list = list(pair_info['video_path'])
        anno_path_list = list(pair_info['anno_path'])


        # inference assets base dir
        inference_assets_base_dir = os.path.join(inference_assets_root_path, video_no)
        
        
        print('\t ==== GETTERING INFO ====')
        print('\t VIDEO NO | ', video_no) 
        print('\t video_path', video_path_list) # target videos path
        print('\t anno_path', anno_path_list) # target annotaion path
        print('\t inference assets base path | ', inference_assets_base_dir)
        print('\t ', glob.glob(inference_assets_base_dir + '/*'))
        print('\t ==== ==== ==== ====\n')

        # it will be append to info_dict
        target_video_list = []
        target_anno_list = []
        target_inference_assets_list = []
        
        for target_video_path, target_anno_path in (zip(video_path_list, anno_path_list)) :
            # init
            temp_inference_assets_list = []
            # it will be append to temp_anno_list
            target_idx_list = []

            # consist infernce assets
            temp_inference_assets_list = glob.glob(os.path.join(inference_assets_base_dir, os.path.splitext(os.path.basename(target_video_path))[0], '*')) # [video1_1_0, video1_1_1, ...]

            # only target_video_path 
            if target_anno_path != '' :
                anno_df = pd.read_csv(target_anno_path)
                anno_df = anno_df.dropna(axis=0) # 결측행 제거

                # time -> frame idx
                for i in range(len(anno_df)) :
                    t_start = anno_df.iloc[i]['start']
                    t_end = anno_df.iloc[i]['end']
                    
                    target_idx_list.append([time_to_idx(t_start, fps), time_to_idx(t_end, fps)]) # temp_idx_list = [[start, end], [start, end]..]
                
                print('-----'*3)
                print('target_video_path \t | ', target_video_path)
                print('inf_assets \t\t |', temp_inference_assets_list)
                print('anno_path \t\t | ', target_anno_path)
                print(anno_df)

            else : # no event
                print('-----'*3)
                print('target_video_path \t | ', target_video_path)
                print('inf_assets \t\t |', temp_inference_assets_list)
                print('anno_path \t\t | ', target_anno_path)
            
            # save gettering info
            target_video_list.append(target_video_path) # [video1_1, video_1_2, ...]
            target_anno_list.append(target_idx_list) # [temp_idx_list_1_1, temp_idx_list_1_2, ... ]
            target_inference_assets_list.append(temp_inference_assets_list) # [video1_1_0, video_1_1_1, video_1_1_2, ..]

        # info_dict['video'], info_dict['anno'] length is same as valset
        info_dict['video'].append(target_video_list) # [[video1_1, video1_2], [video2_1, video_2_2], ...]
        info_dict['anno'].append(target_anno_list) # [[temp_idx_list_1_1, temp_idx_list_1_2], [temp_idx_list_2_1, temp_idx_list_2_2,], ...]
        info_dict['inference_assets'].append(target_inference_assets_list) # [[[video1_1_0, video1_1_1, video1_1_2,..], [video1_2_0, ...]], ... ]
        
        print('\n\n')
        
    return info_dict
    
# 불러온 annotation file 정보가 정확한지 체크 및 수정
def sanity_check_info_dict(info_dict) :
    """
    Generate 'info_dict' (Complete form of 'info_dict') 

    Args:
        info_dict: info_dict from 'gettering_information_for_oob'

    Returns:
        info_dict: The final form of 'info_dict'.
            info_dict = {
            'video': [video1_path, video2_path, ... ],
            'anno': [ 1-[[start, end],[start, end]], 2-[[start,end],[start,end]], 3-... ],
            'inference_assets' : [test_dataset1_path, test_dataset2_path, ...]
            }
    """

    # loop from total_videoset_cnt
    for i, (video_path_list, anno_info_list) in enumerate(zip(info_dict['video'], info_dict['anno']), 0): 
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice = os.path.splitext(video_path_list[0])[0].split('_')
        videoset_name = '{}_{}'.format(op_method, patient_idx)

        for j, (video_path, anno_info) in enumerate(zip(video_path_list, anno_info_list), 0) :
            
            video_name = os.path.splitext(os.path.basename(video_path))[0] # only video name
            print('----- ANNOTATION CHECK => \t VIDEO\t {} \t-----'.format(video_name))
            print(info_dict['anno'][i][j])
            
            ##### video info and ####
            # open video cap for parse frame
            video = cv2.VideoCapture(video_path)
            video_fps = video.get(cv2.CAP_PROP_FPS)
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
            del video

            print('\tTarget video : {} | Total Frame : {} | Video FPS : {} '.format(video_name, video_len, video_fps))
            print('\tAnnotation Info : {}'.format(anno_info))

            ##### annotation sanity check #####
            ### check idx -> time
            if anno_info : # not empty list
                # init 
                over_ret = None
                int_ret = None
                val = None
                
                print('\t BEFORE ANNOTATION => {}\n'.format(anno_info))

                # last frmae annotation check
                over_ret, val = check_anno_over_frame(anno_info, video_len) # over frame이 아닐경우 True, over frame 일 경우 False
                anno_info = anno_info if over_ret else val # update anno_info | over frame일 경우 refined 된 val 값으로 update
                val = None # clean
                
                # check anntation frame is int
                int_ret, val = check_anno_int(anno_info) # int 일 경우 True, int가 아닐경우 False
                anno_info = anno_info if int_ret else val # update anno_info | frame이 int가 아닐경우 모두 int 로 refined 된 값으로 update

                print('\n\t AFTER ANNOTATION => {}'.format(anno_info))

                ##### update redefined annotation info #### 
                info_dict['anno'][i][j] = anno_info
                print(info_dict['anno'][i][j])

            else : # empty
                print(anno_info)
                print('=====> NO EVENT')
            
            print('')

    return info_dict # redefined info_dict
