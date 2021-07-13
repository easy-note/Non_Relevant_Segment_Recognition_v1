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

import math

from pandas import DataFrame as df

import torch
import pytorch_lightning as pl

import re
import copy

import natsort
import yaml

matplotlib.use('Agg')

def time_to_idx(time, fps):
    t_segment = time.split(':')
    # idx = int(t_segment[0]) * 3600 * fps + int(t_segment[1]) * 60 * fps + int(t_segment[2]) 
    idx = (int(t_segment[0]) * 3600 * fps) + (int(t_segment[1]) * 60 * fps) + (int(t_segment[2]) * fps) + int(t_segment[3]) # [h, m, s, frame] 

    return idx

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

    print('\n\n\n\t\t\t ### STARTING DEF [gettering_information_for_oob] ### \n\n')

    info_dict = {
        'video': [],
        'anno': [],
        'inference_assets' : []
    }

    # init
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
        # all_video_path.remove(os.path.join(video_root_path, '01_G_01_L_423_xx0_01.MP4'))
        # all_anno_path.remove(os.path.join(anno_root_path, '01_G_01_L_423_xx0_01_OOB_16.json'))

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

        # extract video only xx0, ch1 # 21.06.10 HG 추가 - xx0, ch1 만 추출
        video_path_df = video_path_df[video_path_df['video_path'].str.contains('|'.join(['xx0_', 'ch1_']))]

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

            # 21.06.10 HG 수정 - LAPA = json, ROBOT = csv parser
            # only target_video_path 
            if target_anno_path != '' : # event

                if mode=='ROBOT' :  # csv
                    anno_df = pd.read_csv(target_anno_path)
                    anno_df = anno_df.dropna(axis=0) # 결측행 제거

                    # time -> frame idx
                    for i in range(len(anno_df)) :
                        t_start = anno_df.iloc[i]['start'] # time
                        t_end = anno_df.iloc[i]['end'] # time
                        
                        target_idx_list.append([time_to_idx(t_start, fps), time_to_idx(t_end, fps)]) # temp_idx_list = [[start, end], [start, end]..]
                    
                
                    print('-----'*3)
                    print('target_video_path \t | ', target_video_path)
                    print('inf_assets \t\t |', temp_inference_assets_list)
                    print('anno_path \t\t | ', target_anno_path)
                    print(anno_df)

                elif mode=='LAPA' : # json
                    with open(target_anno_path) as json_file :
                            json_data = json.load(json_file)

                    # annotation frame
                    for anno_data in json_data['annotations'] :
                        t_start = anno_data['start'] # frame
                        t_end = anno_data['end'] # frame

                        target_idx_list.append([t_start, t_end]) # temp_idx_list = [[start, end], [start, end]..]

                    print('-----'*3)
                    print('target_video_path \t | ', target_video_path)
                    print('inf_assets \t\t |', temp_inference_assets_list)
                    print('anno_path \t\t | ', target_anno_path)
                    print(json_data['annotations'])


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
            'DB': [video1_DB_path, video2_DB_path, ... ],
            'inference_assets' : [test_dataset1_path, test_dataset2_path, ...]
            }
    """

    # loop from total_videoset_cnt
    for i, (video_path_list, anno_info_list) in enumerate(zip(info_dict['video'], info_dict['anno']), 0): 
        # hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice = os.path.splitext(os.path.basename(video_path_list[0]))[0].split('_') # parsing videoset name # 21.06.25 HG 수정, Bug Fix
        # videoset_name = '{}_{}'.format(op_method, patient_idx)

        for j, (video_path, anno_info) in enumerate(zip(video_path_list, anno_info_list), 0) :
            
            video_name = os.path.splitext(os.path.basename(video_path))[0] # only video name
            print('----- ANNOTATION CHECK => \t VIDEO\t {} \t-----'.format(video_name))
            
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

            else : # empty
                print(anno_info)
                print('=====> NO EVENT')
            
            print('')

    return info_dict # redefined info_dict


###### OOB VIDEO LIST ########
# ANNOTATION PATH - /NAS/DATA2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1 # /data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1
# ANNOTATION PATH - /NAS/DATA2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2 # /data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2

# 91ea, 40case # /NAS/DATA/HuToM/Video_Robot_cordname # /data1/HuToM/Video_Robot_cordname
OOB_robot_40 = [
    'R_1_ch1_01', 'R_1_ch1_03', 'R_1_ch1_06', 'R_2_ch1_01', 'R_2_ch1_03', 'R_3_ch1_01', 'R_3_ch1_03', 'R_3_ch1_05', 'R_4_ch1_01', 'R_4_ch1_04', 
    'R_5_ch1_01', 'R_5_ch1_03', 'R_6_ch1_01', 'R_6_ch1_03', 'R_6_ch1_05', 'R_7_ch1_01', 'R_7_ch1_04', 'R_10_ch1_01', 'R_10_ch1_03', 'R_13_ch1_01', 
    'R_13_ch1_03', 'R_14_ch1_01', 'R_14_ch1_03', 'R_14_ch1_05', 'R_15_ch1_01', 'R_15_ch1_03', 'R_17_ch1_01', 'R_17_ch1_04', 'R_17_ch1_06', 'R_18_ch1_01', 
    'R_18_ch1_04', 'R_19_ch1_01', 'R_19_ch1_03', 'R_19_ch1_05', 'R_22_ch1_01', 'R_22_ch1_03', 'R_22_ch1_05', 'R_48_ch1_01', 'R_48_ch1_02', 'R_56_ch1_01', 
    'R_56_ch1_03', 'R_74_ch1_01', 'R_74_ch1_03', 'R_76_ch1_01', 'R_76_ch1_03', 'R_84_ch1_01', 'R_84_ch1_03', 'R_94_ch1_01', 'R_94_ch1_03', 'R_100_ch1_01', 
    'R_100_ch1_03', 'R_100_ch1_05', 'R_116_ch1_01', 'R_116_ch1_03', 'R_116_ch1_06', 'R_117_ch1_01', 'R_117_ch1_03', 'R_201_ch1_01', 'R_201_ch1_03', 'R_202_ch1_01', 
    'R_202_ch1_03', 'R_202_ch1_05', 'R_203_ch1_01', 'R_203_ch1_03', 'R_204_ch1_01', 'R_204_ch1_02', 'R_205_ch1_01', 'R_205_ch1_03', 'R_205_ch1_05', 'R_206_ch1_01', 
    'R_206_ch1_03', 'R_207_ch1_01', 'R_207_ch1_03', 'R_208_ch1_01', 'R_208_ch1_03', 'R_209_ch1_01', 'R_209_ch1_03', 'R_210_ch1_01', 'R_210_ch2_04', 'R_301_ch1_01',
    'R_301_ch1_04', 'R_302_ch1_01', 'R_302_ch1_04', 'R_303_ch1_01', 'R_303_ch1_04', 'R_304_ch1_01', 'R_304_ch1_03', 'R_305_ch1_01', 'R_305_ch1_04', 'R_313_ch1_01', 'R_313_ch1_03']

# 134ea, 60case # /NAS/DATA2/Video/Robot/Dataset2_60case # /data2/Video/Robot/Dataset2_60case
OOB_robot_60 = [
    'R_310_ch1_01', 'R_310_ch1_03', 'R_311_ch1_01', 'R_311_ch1_03', 'R_312_ch1_02', 'R_312_ch1_03', 'R_320_ch1_01', 'R_320_ch1_03', 'R_321_ch1_01', 'R_321_ch1_03', 
    'R_321_ch1_05', 'R_324_ch1_01', 'R_324_ch1_03', 'R_329_ch1_01', 'R_329_ch1_03', 'R_334_ch1_01', 'R_334_ch1_03', 'R_336_ch1_01', 'R_336_ch1_04', 'R_338_ch1_01', 
    'R_338_ch1_03', 'R_338_ch1_05', 'R_339_ch1_01', 'R_339_ch1_03', 'R_339_ch1_05', 'R_340_ch1_01', 'R_340_ch1_03', 'R_340_ch1_05', 'R_342_ch1_01', 'R_342_ch1_03', 
    'R_342_ch1_05', 'R_345_ch1_01', 'R_345_ch1_04', 'R_346_ch1_02', 'R_346_ch1_04', 'R_347_ch1_02', 'R_347_ch1_03', 'R_347_ch1_05', 'R_348_ch1_01', 'R_348_ch1_03', 
    'R_349_ch1_01', 'R_349_ch1_04', 'R_355_ch1_02', 'R_355_ch1_04', 'R_357_ch1_01', 'R_357_ch1_03', 'R_357_ch1_05', 'R_358_ch1_01', 'R_358_ch1_03', 'R_358_ch1_05', 
    'R_362_ch1_01', 'R_362_ch1_03', 'R_362_ch1_05', 'R_363_ch1_01', 'R_363_ch1_03', 'R_369_ch1_01', 'R_369_ch1_03', 'R_372_ch1_01', 'R_372_ch1_04', 'R_376_ch1_01', 
    'R_376_ch1_03', 'R_376_ch1_05', 'R_378_ch1_01', 'R_378_ch1_03', 'R_378_ch1_05', 'R_379_ch1_02', 'R_379_ch1_04', 'R_386_ch1_01', 'R_386_ch1_03', 'R_391_ch1_01', 
    'R_391_ch1_03', 'R_391_ch2_06', 'R_393_ch1_01', 'R_393_ch1_04', 'R_399_ch1_01', 'R_399_ch1_04', 'R_400_ch1_01', 'R_400_ch1_03', 'R_402_ch1_01', 'R_402_ch1_03', 
    'R_403_ch1_01', 'R_403_ch1_03', 'R_405_ch1_01', 'R_405_ch1_03', 'R_405_ch1_05', 'R_406_ch1_02', 'R_406_ch1_04', 'R_406_ch1_06', 'R_409_ch1_01', 'R_409_ch1_03', 
    'R_412_ch1_01', 'R_412_ch1_03', 'R_413_ch1_02', 'R_413_ch1_04', 'R_415_ch1_01', 'R_415_ch1_03', 'R_415_ch1_05', 'R_418_ch1_02', 'R_418_ch1_04', 'R_418_ch1_06', 
    'R_419_ch1_01', 'R_419_ch1_04', 'R_420_ch1_01', 'R_420_ch1_03', 'R_423_ch1_01', 'R_423_ch1_03', 'R_424_ch2_02', 'R_424_ch2_04', 'R_427_ch1_01', 'R_427_ch1_03', 
    'R_436_ch1_02', 'R_436_ch1_04', 'R_436_ch1_06', 'R_436_ch1_08', 'R_436_ch1_10', 'R_445_ch1_01', 'R_445_ch1_03', 'R_449_ch1_01', 'R_449_ch1_04', 'R_449_ch1_06', 
    'R_455_ch1_01', 'R_455_ch1_03', 'R_455_ch1_05', 'R_480_ch1_01', 'R_493_ch1_01', 'R_493_ch1_03', 'R_501_ch1_01', 'R_510_ch1_01', 'R_510_ch1_03', 'R_522_ch1_01', 
    'R_523_ch1_01', 'R_526_ch1_01', 'R_532_ch1_01', 'R_533_ch1_01']

# 225ea & 100case 
OOB_robot_list = OOB_robot_40 + OOB_robot_60

# 350ea, 40case # /NAS/DATA2/Public/IDC_21.06.25/Dataset1 # /data2/Public/IDC_21.06.25/Dataset1
OOB_lapa_40 = [
    'L_301_xx0_01', 'L_301_xx0_02', 'L_301_xx0_03', 'L_301_xx0_04', 'L_301_xx0_05', 'L_301_xx0_06', 'L_303_xx0_01', 'L_303_xx0_02', 'L_303_xx0_03', 'L_303_xx0_04', 
    'L_303_xx0_05', 'L_303_xx0_06', 'L_305_xx0_01', 'L_305_xx0_02', 'L_305_xx0_03', 'L_305_xx0_04', 'L_305_xx0_05', 'L_305_xx0_06', 'L_305_xx0_07', 'L_305_xx0_08', 
    'L_305_xx0_09', 'L_305_xx0_10', 'L_305_xx0_11', 'L_305_xx0_12', 'L_305_xx0_13', 'L_305_xx0_14', 'L_305_xx0_15', 'L_309_xx0_01', 'L_309_xx0_02', 'L_309_xx0_03', 
    'L_309_xx0_04', 'L_309_xx0_05', 'L_309_xx0_06', 'L_309_xx0_07', 'L_317_xx0_01', 'L_317_xx0_02', 'L_317_xx0_03', 'L_317_xx0_04', 'L_325_xx0_01', 'L_325_xx0_02', 
    'L_325_xx0_03', 'L_325_xx0_04', 'L_325_xx0_05', 'L_325_xx0_06', 'L_325_xx0_07', 'L_325_xx0_08', 'L_325_xx0_09', 'L_325_xx0_10', 'L_325_xx0_11', 'L_325_xx0_12', 
    'L_326_xx0_01', 'L_326_xx0_02', 'L_326_xx0_03', 'L_326_xx0_04', 'L_326_xx0_05', 'L_326_xx0_06', 'L_340_xx0_01', 'L_340_xx0_02', 'L_340_xx0_03', 'L_340_xx0_04', 
    'L_340_xx0_05', 'L_340_xx0_06', 'L_340_xx0_07', 'L_340_xx0_08', 'L_340_xx0_09', 'L_340_xx0_10', 'L_346_xx0_01', 'L_346_xx0_02', 'L_349_ch1_01', 'L_349_ch1_02', 
    'L_349_ch1_03', 'L_349_ch1_04', 'L_412_xx0_01', 'L_412_xx0_02', 'L_412_xx0_03', 'L_421_xx0_01', 'L_421_xx0_02', 'L_423_xx0_01', 'L_423_xx0_02', 'L_423_xx0_03', 
    'L_423_xx0_04', 'L_423_xx0_05', 'L_442_xx0_01', 'L_442_xx0_02', 'L_442_xx0_03', 'L_442_xx0_04', 'L_442_xx0_05', 'L_442_xx0_06', 'L_442_xx0_07', 'L_442_xx0_08', 
    'L_442_xx0_09', 'L_442_xx0_10', 'L_442_xx0_11', 'L_442_xx0_12', 'L_442_xx0_13', 'L_442_xx0_14', 'L_443_xx0_01', 'L_443_xx0_02', 'L_443_xx0_03', 'L_443_xx0_04', 
    'L_443_xx0_05', 'L_443_xx0_06', 'L_443_xx0_07', 'L_443_xx0_08', 'L_443_xx0_09', 'L_443_xx0_10', 'L_443_xx0_11', 'L_443_xx0_12', 'L_443_xx0_13', 'L_443_xx0_14', 
    'L_443_xx0_15', 'L_443_xx0_16', 'L_450_xx0_01', 'L_450_xx0_02', 'L_450_xx0_03', 'L_450_xx0_04', 'L_450_xx0_05', 'L_450_xx0_06', 'L_450_xx0_07', 'L_450_xx0_08', 
    'L_450_xx0_09', 'L_450_xx0_10', 'L_450_xx0_11', 'L_450_xx0_12', 'L_450_xx0_13', 'L_450_xx0_14', 'L_450_xx0_15', 'L_450_xx0_16', 'L_450_xx0_17', 'L_450_xx0_18', 
    'L_450_xx0_19', 'L_450_xx0_20', 'L_450_xx0_21', 'L_450_xx0_22', 'L_458_xx0_01', 'L_458_xx0_02', 'L_458_xx0_03', 'L_458_xx0_04', 'L_458_xx0_05', 'L_458_xx0_06', 
    'L_458_xx0_07', 'L_458_xx0_08', 'L_458_xx0_09', 'L_458_xx0_10', 'L_458_xx0_11', 'L_458_xx0_12', 'L_458_xx0_13', 'L_458_xx0_14', 'L_458_xx0_15', 'L_465_xx0_01', 
    'L_465_xx0_02', 'L_465_xx0_03', 'L_465_xx0_04', 'L_465_xx0_05', 'L_465_xx0_06', 'L_465_xx0_07', 'L_465_xx0_08', 'L_465_xx0_09', 'L_465_xx0_10', 'L_465_xx0_11', 
    'L_465_xx0_12', 'L_465_xx0_13', 'L_465_xx0_14', 'L_465_xx0_15', 'L_465_xx0_16', 'L_465_xx0_17', 'L_465_xx0_18', 'L_465_xx0_19', 'L_465_xx0_20', 'L_465_xx0_21', 
    'L_491_xx0_01', 'L_491_xx0_02', 'L_491_xx0_03', 'L_491_xx0_04', 'L_491_xx0_05', 'L_491_xx0_06', 'L_491_xx0_07', 'L_491_xx0_08', 'L_491_xx0_09', 'L_491_xx0_10', 
    'L_491_xx0_11', 'L_491_xx0_12', 'L_493_ch1_01', 'L_493_ch1_02', 'L_493_ch1_03', 'L_493_ch1_04', 'L_496_ch1_01', 'L_496_ch1_02', 'L_496_ch1_03', 'L_507_xx0_01', 
    'L_507_xx0_02', 'L_507_xx0_03', 'L_507_xx0_04', 'L_507_xx0_05', 'L_507_xx0_06', 'L_507_xx0_07', 'L_522_xx0_01', 'L_522_xx0_02', 'L_522_xx0_03', 'L_522_xx0_04', 
    'L_522_xx0_05', 'L_522_xx0_06', 'L_522_xx0_07', 'L_522_xx0_08', 'L_522_xx0_09', 'L_522_xx0_10', 'L_522_xx0_11', 'L_534_xx0_01', 'L_534_xx0_02', 'L_534_xx0_03', 
    'L_534_xx0_04', 'L_534_xx0_05', 'L_534_xx0_06', 'L_534_xx0_07', 'L_535_xx0_01', 'L_535_xx0_02', 'L_535_xx0_03', 'L_535_xx0_04', 'L_535_xx0_05', 'L_550_xx0_01', 
    'L_550_xx0_02', 'L_550_xx0_03', 'L_550_xx0_04', 'L_550_xx0_05', 'L_550_xx0_06', 'L_550_xx0_07', 'L_550_xx0_08', 'L_550_xx0_09', 'L_550_xx0_10', 'L_550_xx0_11', 
    'L_550_xx0_12', 'L_553_ch1_01', 'L_553_ch1_02', 'L_553_ch1_03', 'L_553_ch1_04', 'L_586_xx0_01', 'L_586_xx0_02', 'L_586_xx0_03', 'L_586_xx0_04', 'L_586_xx0_05', 
    'L_586_xx0_06', 'L_586_xx0_07', 'L_586_xx0_08', 'L_586_xx0_09', 'L_586_xx0_10', 'L_586_xx0_11', 'L_586_xx0_12', 'L_586_xx0_13', 'L_586_xx0_14', 'L_586_xx0_15', 
    'L_586_xx0_16', 'L_586_xx0_17', 'L_586_xx0_18', 'L_586_xx0_19', 'L_586_xx0_20', 'L_595_xx0_01', 'L_595_xx0_02', 'L_595_xx0_03', 'L_595_xx0_04', 'L_595_xx0_05', 
    'L_595_xx0_06', 'L_595_xx0_07', 'L_595_xx0_08', 'L_605_xx0_01', 'L_605_xx0_02', 'L_605_xx0_03', 'L_605_xx0_04', 'L_605_xx0_05', 'L_605_xx0_06', 'L_605_xx0_07', 
    'L_605_xx0_08', 'L_605_xx0_09', 'L_605_xx0_10', 'L_605_xx0_11', 'L_605_xx0_12', 'L_605_xx0_13', 'L_605_xx0_14', 'L_605_xx0_15', 'L_605_xx0_16', 'L_605_xx0_17', 
    'L_605_xx0_18', 'L_607_xx0_01', 'L_607_xx0_02', 'L_607_xx0_03', 'L_607_xx0_04', 'L_625_xx0_01', 'L_625_xx0_02', 'L_625_xx0_03', 'L_625_xx0_04', 'L_625_xx0_05', 
    'L_625_xx0_06', 'L_625_xx0_07', 'L_625_xx0_08', 'L_625_xx0_09', 'L_631_xx0_01', 'L_631_xx0_02', 'L_631_xx0_03', 'L_631_xx0_04', 'L_631_xx0_05', 'L_631_xx0_06', 
    'L_631_xx0_07', 'L_631_xx0_08', 'L_647_xx0_01', 'L_647_xx0_02', 'L_647_xx0_03', 'L_647_xx0_04', 'L_654_xx0_01', 'L_654_xx0_02', 'L_654_xx0_03', 'L_654_xx0_04', 
    'L_654_xx0_05', 'L_654_xx0_06', 'L_654_xx0_07', 'L_654_xx0_08', 'L_654_xx0_09', 'L_654_xx0_10', 'L_654_xx0_11', 'L_659_xx0_01', 'L_659_xx0_02', 'L_659_xx0_03', 
    'L_659_xx0_04', 'L_659_xx0_05', 'L_660_xx0_01', 'L_660_xx0_02', 'L_660_xx0_03', 'L_660_xx0_04', 'L_661_xx0_01', 'L_661_xx0_02', 'L_661_xx0_03', 'L_661_xx0_04', 
    'L_661_xx0_05', 'L_661_xx0_06', 'L_661_xx0_07', 'L_661_xx0_08', 'L_661_xx0_09', 'L_661_xx0_10', 'L_661_xx0_11', 'L_661_xx0_12', 'L_661_xx0_13', 'L_661_xx0_14', 
    'L_661_xx0_15', 'L_669_xx0_01', 'L_669_xx0_02', 'L_669_xx0_03', 'L_669_xx0_04', 'L_676_xx0_01', 'L_676_xx0_02', 'L_676_xx0_03', 'L_676_xx0_04', 'L_676_xx0_05']

# 521ea, 60case # /NAS/DATA2/Public/IDC_21.06.25/Dataset2 # /data2/Public/IDC_21.06.25/Dataset2
OOB_lapa_60 = [
    'L_310_xx0_01', 'L_310_xx0_02', 'L_310_xx0_03', 'L_310_xx0_04', 'L_310_xx0_05', 'L_310_xx0_06', 'L_310_xx0_07', 'L_310_xx0_08', 'L_310_xx0_09', 'L_310_xx0_10', 
    'L_310_xx0_11', 'L_310_xx0_12', 'L_311_xx0_01', 'L_311_xx0_02', 'L_311_xx0_03', 'L_311_xx0_04', 'L_311_xx0_05', 'L_330_ch1_01', 'L_333_xx0_01', 'L_333_xx0_02', 
    'L_333_xx0_03', 'L_333_xx0_04', 'L_333_xx0_05', 'L_333_xx0_06', 'L_333_xx0_07', 'L_333_xx0_08', 'L_333_xx0_09', 'L_333_xx0_10', 'L_333_xx0_11', 'L_367_ch1_01', 
    'L_370_ch1_01', 'L_377_ch1_01', 'L_379_xx0_01', 'L_379_xx0_02', 'L_379_xx0_03', 'L_379_xx0_04', 'L_379_xx0_05', 'L_379_xx0_06', 'L_379_xx0_07', 'L_379_xx0_08', 
    'L_379_xx0_09', 'L_379_xx0_10', 'L_379_xx0_11', 'L_385_xx0_01', 'L_385_xx0_02', 'L_385_xx0_03', 'L_385_xx0_04', 'L_385_xx0_05', 'L_385_xx0_06', 'L_385_xx0_07', 
    'L_385_xx0_08', 'L_385_xx0_09', 'L_385_xx0_10', 'L_385_xx0_11', 'L_385_xx0_12', 'L_385_xx0_13', 'L_385_xx0_14', 'L_385_xx0_15', 'L_387_xx0_01', 'L_387_xx0_02', 
    'L_387_xx0_03', 'L_387_xx0_04', 'L_387_xx0_05', 'L_387_xx0_06', 'L_387_xx0_07', 'L_387_xx0_08', 'L_389_xx0_01', 'L_389_xx0_02', 'L_389_xx0_03', 'L_389_xx0_04', 
    'L_389_xx0_05', 'L_389_xx0_06', 'L_389_xx0_07', 'L_389_xx0_08', 'L_389_xx0_09', 'L_389_xx0_10', 'L_389_xx0_11', 'L_389_xx0_12', 'L_389_xx0_13', 'L_391_xx0_01', 
    'L_391_xx0_02', 'L_391_xx0_03', 'L_391_xx0_04', 'L_391_xx0_05', 'L_391_xx0_06', 'L_391_xx0_07', 'L_391_xx0_08', 'L_391_xx0_09', 'L_393_xx0_01', 'L_393_xx0_02', 
    'L_393_xx0_03', 'L_393_xx0_04', 'L_393_xx0_05', 'L_393_xx0_06', 'L_393_xx0_07', 'L_393_xx0_08', 'L_393_xx0_09', 'L_393_xx0_10', 'L_400_xx0_01', 'L_400_xx0_02', 
    'L_400_xx0_03', 'L_400_xx0_04', 'L_400_xx0_05', 'L_400_xx0_06', 'L_400_xx0_07', 'L_400_xx0_08', 'L_400_xx0_09', 'L_400_xx0_10', 'L_400_xx0_11', 'L_400_xx0_12', 
    'L_402_xx0_01', 'L_402_xx0_02', 'L_402_xx0_03', 'L_402_xx0_04', 'L_406_xx0_01', 'L_406_xx0_02', 'L_406_xx0_03', 'L_406_xx0_04', 'L_406_xx0_05', 'L_406_xx0_06', 
    'L_406_xx0_07', 'L_406_xx0_08', 'L_406_xx0_09', 'L_406_xx0_10', 'L_406_xx0_11', 'L_406_xx0_12', 'L_406_xx0_13', 'L_408_ch1_01', 'L_413_xx0_01', 'L_413_xx0_02', 
    'L_413_xx0_03', 'L_413_xx0_04', 'L_413_xx0_05', 'L_413_xx0_06', 'L_413_xx0_07', 'L_413_xx0_08', 'L_413_xx0_09', 'L_413_xx0_10', 'L_414_xx0_01', 'L_414_xx0_02', 
    'L_414_xx0_03', 'L_414_xx0_04', 'L_414_xx0_05', 'L_414_xx0_06', 'L_414_xx0_07', 'L_414_xx0_08', 'L_415_xx0_01', 'L_415_xx0_02', 'L_415_xx0_03', 'L_415_xx0_04', 
    'L_415_xx0_05', 'L_415_xx0_06', 'L_415_xx0_07', 'L_415_xx0_08', 'L_415_xx0_09', 'L_415_xx0_10', 'L_415_xx0_11', 'L_415_xx0_12', 'L_418_xx0_01', 'L_418_xx0_02', 
    'L_418_xx0_03', 'L_418_xx0_04', 'L_418_xx0_05', 'L_418_xx0_06', 'L_418_xx0_07', 'L_418_xx0_08', 'L_419_xx0_01', 'L_419_xx0_02', 'L_419_xx0_03', 'L_419_xx0_04', 
    'L_419_xx0_05', 'L_419_xx0_06', 'L_427_xx0_01', 'L_427_xx0_02', 'L_427_xx0_03', 'L_427_xx0_04', 'L_427_xx0_05', 'L_427_xx0_06', 'L_427_xx0_07', 'L_427_xx0_08', 
    'L_427_xx0_09', 'L_427_xx0_10', 'L_427_xx0_11', 'L_427_xx0_12', 'L_427_xx0_13', 'L_427_xx0_14', 'L_427_xx0_15', 'L_428_ch1_01', 'L_430_ch1_01', 'L_433_xx0_01', 
    'L_433_xx0_02', 'L_433_xx0_03', 'L_433_xx0_04', 'L_433_xx0_05', 'L_433_xx0_06', 'L_433_xx0_07', 'L_433_xx0_08', 'L_433_xx0_09', 'L_434_xx0_01', 'L_434_xx0_02', 
    'L_434_xx0_03', 'L_434_xx0_04', 'L_434_xx0_05', 'L_434_xx0_06', 'L_434_xx0_07', 'L_434_xx0_08', 'L_434_xx0_09', 'L_434_xx0_10', 'L_436_xx0_01', 'L_436_xx0_02', 
    'L_436_xx0_03', 'L_436_xx0_04', 'L_436_xx0_05', 'L_436_xx0_06', 'L_436_xx0_07', 'L_436_xx0_08', 'L_436_xx0_09', 'L_436_xx0_10', 'L_436_xx0_11', 'L_436_xx0_12', 
    'L_439_xx0_01', 'L_439_xx0_02', 'L_439_xx0_03', 'L_439_xx0_04', 'L_439_xx0_05', 'L_439_xx0_06', 'L_439_xx0_07', 'L_439_xx0_08', 'L_439_xx0_09', 'L_439_xx0_10', 
    'L_439_xx0_11', 'L_439_xx0_12', 'L_439_xx0_13', 'L_439_xx0_14', 'L_439_xx0_15', 'L_439_xx0_16', 'L_471_xx0_01', 'L_471_xx0_02', 'L_471_xx0_03', 'L_471_xx0_04', 
    'L_471_xx0_05', 'L_471_xx0_06', 'L_471_xx0_07', 'L_471_xx0_08', 'L_471_xx0_09', 'L_471_xx0_10', 'L_471_xx0_11', 'L_473_xx0_01', 'L_473_xx0_02', 'L_473_xx0_03', 
    'L_473_xx0_04', 'L_473_xx0_05', 'L_473_xx0_06', 'L_473_xx0_07', 'L_475_ch1_01', 'L_475_ch1_02', 'L_477_ch1_01', 'L_478_xx0_01', 'L_478_xx0_02', 'L_478_xx0_03', 
    'L_478_xx0_04', 'L_478_xx0_05', 'L_478_xx0_06', 'L_478_xx0_07', 'L_478_xx0_08', 'L_478_xx0_09', 'L_478_xx0_10', 'L_479_xx0_01', 'L_479_xx0_02', 'L_479_xx0_03', 
    'L_479_xx0_04', 'L_479_xx0_05', 'L_479_xx0_06', 'L_479_xx0_07', 'L_479_xx0_08', 'L_479_xx0_09', 'L_481_xx0_01', 'L_481_xx0_02', 'L_481_xx0_03', 'L_481_xx0_04', 
    'L_481_xx0_05', 'L_481_xx0_06', 'L_481_xx0_07', 'L_481_xx0_08', 'L_481_xx0_09', 'L_481_xx0_10', 'L_481_xx0_11', 'L_481_xx0_12', 'L_481_xx0_13', 'L_482_xx0_01', 
    'L_482_xx0_02', 'L_482_xx0_03', 'L_482_xx0_04', 'L_482_xx0_05', 'L_482_xx0_06', 'L_482_xx0_07', 'L_482_xx0_08', 'L_482_xx0_09', 'L_482_xx0_10', 'L_482_xx0_11', 
    'L_482_xx0_12', 'L_482_xx0_13', 'L_482_xx0_14', 'L_482_xx0_15', 'L_484_xx0_01', 'L_484_xx0_02', 'L_484_xx0_03', 'L_484_xx0_04', 'L_484_xx0_05', 'L_484_xx0_06', 
    'L_484_xx0_07', 'L_484_xx0_08', 'L_484_xx0_09', 'L_484_xx0_10', 'L_484_xx0_11', 'L_513_xx0_01', 'L_513_xx0_02', 'L_513_xx0_03', 'L_513_xx0_04', 'L_513_xx0_05', 
    'L_513_xx0_06', 'L_513_xx0_07', 'L_513_xx0_08', 'L_513_xx0_09', 'L_513_xx0_10', 'L_513_xx0_11', 'L_513_xx0_12', 'L_514_xx0_01', 'L_514_xx0_02', 'L_514_xx0_03', 
    'L_514_xx0_04', 'L_514_xx0_05', 'L_514_xx0_06', 'L_514_xx0_07', 'L_514_xx0_08', 'L_514_xx0_09', 'L_514_xx0_10', 'L_515_xx0_01', 'L_515_xx0_02', 'L_515_xx0_03', 
    'L_515_xx0_04', 'L_515_xx0_05', 'L_515_xx0_06', 'L_515_xx0_07', 'L_515_xx0_08', 'L_517_xx0_01', 'L_517_xx0_02', 'L_517_xx0_03', 'L_517_xx0_04', 'L_517_xx0_05', 
    'L_517_xx0_06', 'L_517_xx0_07', 'L_517_xx0_08', 'L_537_xx0_01', 'L_537_xx0_02', 'L_537_xx0_03', 'L_537_xx0_04', 'L_537_xx0_05', 'L_537_xx0_06', 'L_537_xx0_07', 
    'L_537_xx0_08', 'L_537_xx0_09', 'L_537_xx0_10', 'L_537_xx0_11', 'L_537_xx0_12', 'L_539_ch1_01', 'L_542_xx0_01', 'L_542_xx0_02', 'L_542_xx0_03', 'L_542_xx0_04', 
    'L_542_xx0_05', 'L_542_xx0_06', 'L_542_xx0_07', 'L_542_xx0_08', 'L_542_xx0_09', 'L_543_xx0_01', 'L_543_xx0_02', 'L_543_xx0_03', 'L_543_xx0_04', 'L_543_xx0_05', 
    'L_543_xx0_06', 'L_543_xx0_07', 'L_543_xx0_08', 'L_543_xx0_09', 'L_543_xx0_10', 'L_543_xx0_11', 'L_543_xx0_12', 'L_543_xx0_13', 'L_543_xx0_14', 'L_545_xx0_01', 
    'L_545_xx0_02', 'L_545_xx0_03', 'L_545_xx0_04', 'L_545_xx0_05', 'L_545_xx0_06', 'L_545_xx0_07', 'L_545_xx0_08', 'L_545_xx0_09', 'L_546_xx0_01', 'L_546_xx0_02', 
    'L_546_xx0_03', 'L_546_xx0_04', 'L_546_xx0_05', 'L_546_xx0_06', 'L_546_xx0_07', 'L_546_xx0_08', 'L_546_xx0_09', 'L_546_xx0_10', 'L_556_xx0_01', 'L_556_xx0_02', 
    'L_556_xx0_03', 'L_556_xx0_04', 'L_556_xx0_05', 'L_556_xx0_06', 'L_556_xx0_07', 'L_556_xx0_08', 'L_556_xx0_09', 'L_556_xx0_10', 'L_556_xx0_11', 'L_556_xx0_12', 
    'L_556_xx0_13', 'L_556_xx0_14', 'L_558_xx0_01', 'L_560_xx0_01', 'L_560_xx0_02', 'L_560_xx0_03', 'L_560_xx0_04', 'L_560_xx0_05', 'L_560_xx0_06', 'L_560_xx0_07', 
    'L_560_xx0_08', 'L_560_xx0_09', 'L_560_xx0_10', 'L_560_xx0_11', 'L_560_xx0_12', 'L_560_xx0_13', 'L_560_xx0_14', 'L_560_xx0_15', 'L_563_xx0_01', 'L_563_xx0_02', 
    'L_563_xx0_03', 'L_563_xx0_04', 'L_563_xx0_05', 'L_563_xx0_06', 'L_563_xx0_07', 'L_563_xx0_08', 'L_563_xx0_09', 'L_563_xx0_10', 'L_563_xx0_11', 'L_563_xx0_12', 
    'L_565_xx0_01', 'L_565_xx0_02', 'L_565_xx0_03', 'L_565_xx0_04', 'L_565_xx0_05', 'L_565_xx0_06', 'L_565_xx0_07', 'L_565_xx0_08', 'L_565_xx0_09', 'L_568_xx0_01', 
    'L_568_xx0_02', 'L_568_xx0_03', 'L_568_xx0_04', 'L_568_xx0_05', 'L_568_xx0_06', 'L_568_xx0_07', 'L_568_xx0_08', 'L_568_xx0_09', 'L_569_xx0_01', 'L_569_xx0_02', 
    'L_569_xx0_03', 'L_569_xx0_04', 'L_569_xx0_05', 'L_569_xx0_06', 'L_569_xx0_07', 'L_569_xx0_08', 'L_569_xx0_09', 'L_569_xx0_10', 'L_569_xx0_11', 'L_569_xx0_12', 
    'L_572_ch1_01', 'L_574_xx0_01', 'L_574_xx0_02', 'L_574_xx0_03', 'L_574_xx0_04', 'L_574_xx0_05', 'L_574_xx0_06', 'L_574_xx0_07', 'L_574_xx0_08', 'L_574_xx0_09', 
    'L_574_xx0_10', 'L_574_xx0_11', 'L_575_xx0_01', 'L_575_xx0_02', 'L_575_xx0_03', 'L_575_xx0_04', 'L_577_xx0_01', 'L_577_xx0_02', 'L_577_xx0_03', 'L_577_xx0_04', 
    'L_577_xx0_05', 'L_577_xx0_06', 'L_577_xx0_07', 'L_577_xx0_08', 'L_577_xx0_09', 'L_577_xx0_10', 'L_580_xx0_01', 'L_580_xx0_02', 'L_580_xx0_03', 'L_580_xx0_04', 
    'L_580_xx0_05', 'L_580_xx0_06', 'L_580_xx0_07', 'L_580_xx0_08', 'L_580_xx0_09', 'L_580_xx0_10', 'L_580_xx0_11', 'L_580_xx0_12', 'L_580_xx0_13', 'L_580_xx0_14', 'L_580_xx0_15']

# 871ea, 100case
OOB_lapa_list = OOB_lapa_40 + OOB_lapa_60

## OLD NAS POLICY
# OLB : R000001/ch1_video_01.mp4
# NEW : R_1_ch1_01.mp4
# DB : R_1/01_G_01_R_1_ch1_01/01_G_01_R_1_ch1_01-0000000001.jpg

# Parsing => Change to new policy => use video Filtering

'''
[ONLY FOR ROBOT]
convert to name for new nas policy from old nas policy
- old_video_path = ~/R000001/ch1_video_01.mp4
'''
def convert_video_name_from_old_nas_policy(old_video_path) : # ONLY FOR ROBOT
    parents_dir, video = old_video_path.split(os.path.sep)[-2:] # R000001, ch1_video_01.mp4
    video_name, ext = os.path.splitext(video) # ch1_video_01, .mp4

    OP_method, patient_no = re.findall(r'R|\d+', parents_dir) # R, 000001
    
    patient_no = str(int(patient_no)) # 000001 => 1

    video_ch, _, video_no = video_name.split('_') # ch1, video, 01

    new_nas_policy_name = "_".join([OP_method, patient_no, video_ch, video_no]) # R_1_ch1_01
    # print('CONVERTED : {} \t ===> \t {}'.format(old_video_path, new_nas_policy_name))

    return new_nas_policy_name
    

# FOR OLD NAS POLICY VIDEO PATH (ROBOT)
'''
video_root_path : video_root_path/Rxxxxxx/ch1_video_01.mp4
video_list = [R_1_ch1_01, ]
'''
def get_video_path_for_robot(video_root_path, video_list) :
    print('\n{}'.format('=====' * 10))
    print('\t ===== GET VIDEO PATH FOR ROBOT =====')
    print('{}\n'.format('=====' * 10))
    

    # DATA_PATH_DICT
    video_path_dict = {}

    # set for video use list, should copy because of list remove
    USE_VIDEO_LIST = video_list.copy()
    
    # video_root_path/R000001/ch2_video_02.mp4 ==> only parsing this rule | RXXXXXX/\w{3}_video_\d+[.]MP4, mp4
    base_video_path = glob.glob(os.path.join(video_root_path, '*', '*')) # total path
    
    parser_regex = re.compile(r'[R]\d+/\w{3}_video_\d+[.]mp4$|[R]\d+/\w{3}_video_\d+[.]MP4$') # R000001/ch2_video_02.mp4
    pattern = re.compile(parser_regex)
    
    cnt = 0 # PROCESSED CNT
    
    # parsering with regex
    for path in base_video_path : 
        match = pattern.search(path)
        if match : # MATCHED
            # print('Matching \t', match.group())
            video_name = convert_video_name_from_old_nas_policy(path)
            
            if video_name in USE_VIDEO_LIST : # Only Processing in USE VIDEO LIST
                # print('USE VIDEO : {} \n'.format(video_name))
                video_path_dict[video_name] = path # add dict | new_nas_video_name : video_path
                USE_VIDEO_LIST.remove(video_name)
                cnt+=1
                

        else : # UNMATCHED
            pass
            # print('NON Matching \t', path)


    print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(USE_VIDEO_LIST)))
    print('FALIED DATA LIST : {}'.format(USE_VIDEO_LIST))

    ### ADD EXCEPTION FILE
    EXCEPTION_RULE = {'R_76_ch1_01' : os.path.join(video_root_path, 'R000076', 'ch1_video_01_6915320_RDG.mp4'),
                       'R_84_ch1_01' : os.path.join(video_root_path, 'R000084', 'ch1_video_01_8459178_robotic\ subtotal.mp4'), 
                       'R_391_ch2_06' : os.path.join(video_root_path, 'R000391', '01_G_01_R_391_ch2_06.mp4')}

    print('\n--REGISTERD EXCEPTION DATA RULE --\n')
    for keys, value in EXCEPTION_RULE.items() : 
        print('{} | {}'.format(keys, value))

    print('\n ==> APPLY EXEPTION RULE')
    
    # apply EXCEPTION RULE
    for video_name in USE_VIDEO_LIST :
        video_path_dict[video_name] = EXCEPTION_RULE.get(video_name, '') # if Non key, return ''
        print('{} | {}'.format(video_name, video_path_dict[video_name]))

    print('\n----- RESULTS [VIDEO_PATH_DICT CNT : {} ] ------\n'.format(len(video_path_dict)))
    for keys, value in video_path_dict.items() : 
        print('{} | {}'.format(keys, value))

    return video_path_dict


# FOR NEW NAS POLICY VIDEO PATH
def get_video_path(video_root_path, video_list) :
    print('\n{}'.format('=====' * 10))
    print('\t ===== GET VIDEO PATH =====')
    print('{}\n'.format('=====' * 10))
    
    # DATA_PATH_DICT
    video_path_dict = {}

    # set for video use list, should copy because of list remove
    USE_VIDEO_LIST = video_list.copy()
    FALID_VIDEO_LIST = video_list.copy() # pop when matched

    # parsing all video path
    all_video_path = []
    video_ext_list = ['mp4', 'MP4', 'mpg']

    for ext in video_ext_list :
        all_video_path.extend(glob.glob(os.path.join(video_root_path, '*.{}'.format(ext))))
    
    cnt = 0 # PROCESSED CNT

    # check which idx is included parser_str in all_video_path
    for video_name in USE_VIDEO_LIST :
        idx = return_idx_is_str_in_list(video_name, all_video_path)
        
        if idx == -1 :
            video_path_dict[video_name] = ''
        else : 
            cnt += 1
            video_path_dict[video_name] = all_video_path[idx]
            FALID_VIDEO_LIST.remove(video_name)

    print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(FALID_VIDEO_LIST)))
    print('FALIED DATA LIST : {}'.format(FALID_VIDEO_LIST))

    ### ADD EXCEPTION FILE
    EXCEPTION_RULE = {}

    print('\n--REGISTERD EXCEPTION DATA RULE --\n')
    for keys, value in EXCEPTION_RULE.items() : 
        print('{} | {}'.format(keys, value))

    print('\n ==> APPLY EXEPTION RULE')
    
    # apply EXCEPTION RULE
    for video_name in FALID_VIDEO_LIST :
        video_path_dict[video_name] = EXCEPTION_RULE.get(video_name, '') # if Non key, return ''
        print('{} | {}'.format(video_name, video_path_dict[video_name]))

    print('\n----- RESULTS [VIDEO_PATH_DICT CNT : {} ] ------\n'.format(len(video_path_dict)))
    for keys, value in video_path_dict.items() : 
        print('{} | {}'.format(keys, value))

    return video_path_dict

def get_anno_path(anno_root_path, video_list) :
    print('\n{}'.format('=====' * 10))
    print('\t ===== GET ANNOTATION PATH =====')
    print('{}\n'.format('=====' * 10))

    # DATA_PATH_DICT
    anno_path_dict = {}

    # set USE VIDEO LIST
    USE_VIDEO_LIST = video_list.copy()
    
    # parsing all annotation path
    all_anno_path = []
    anno_ext_list = ['json']

    for ext in anno_ext_list :
        all_anno_path.extend(glob.glob(os.path.join(anno_root_path, '*.{}'.format(ext))))

    cnt = 0 # PROCESSED CNT

    # check which idx is included parser_str in all_anno_path
    for video_name in USE_VIDEO_LIST :
        idx = return_idx_is_str_in_list(video_name, all_anno_path)
        
        if idx == -1 :
            anno_path_dict[video_name] = ''
        else : 
            cnt += 1
            anno_path_dict[video_name] = all_anno_path[idx]

    print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(USE_VIDEO_LIST)-cnt))
    # print('FALIED DATA LIST : {}'.format(USE_VIDEO_LIST))
    
    print('\n----- RESULTS [ANNO_PATH_DICT CNT : {} ] ------\n'.format(len(anno_path_dict)))
    for keys, value in anno_path_dict.items() : 
        print('{} | {}'.format(keys, value))

    return anno_path_dict


# return index of target_list which has parser_str, if not return -1
def return_idx_is_str_in_list(parser_str, target_list) :
    find_idx = -1 # EXCEPTION NUM
    
    for idx, target in enumerate(target_list) :
        if parser_str in target :
            find_idx = idx
            break

    return find_idx

# return elements of target_list which has parser_str, if not return []
def return_elements_is_str_in_list(parser_str, target_list) :
    find_elements = [] # Non
    
    for idx, target in enumerate(target_list) :
        if parser_str in target :
            find_elements.append(target)

    return find_elements


def get_DB_path(DB_root_path, video_list) :
    print('\n{}'.format('=====' * 10))
    print('\t ===== GET DB PATH =====')
    print('{}\n'.format('=====' * 10))

    # DATA_PATH_DICT
    DB_path_dict = {}

    # set USE VIDEO LIST
    USE_VIDEO_LIST = video_list.copy()

    # parsing only DIRECTORY in DB root path
    all_DB_path = [path for path in glob.glob(os.path.join(DB_root_path, '*', '*'), recursive=False) if os.path.isdir(path)] # DB_root_path / Patient / Video_name / Video_name-000000001.jpg

    cnt = 0 # PROCESSED CNT

    # check which idx is included parser_str in all_anno_path
    for video_name in USE_VIDEO_LIST :
        idx = return_idx_is_str_in_list(video_name, all_DB_path)
        
        if idx == -1 :
            DB_path_dict[video_name] = ''
        else : 
            cnt += 1
            DB_path_dict[video_name] = all_DB_path[idx]

    print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(USE_VIDEO_LIST)-cnt))
    
    print('\n----- RESULTS [ANNO_PATH_DICT CNT : {} ] ------\n'.format(len(DB_path_dict)))
    for keys, value in DB_path_dict.items() : 
        print('{} | {}'.format(keys, value))

    return DB_path_dict


# parsing patient video from video_list 
def parsing_patient_video(patient_list, video_list) : 
    print('\n{}'.format('=====' * 10))
    print('\t ===== GET PATIENT VIDEO ===== \t')
    print('{}\n'.format('=====' * 10))

    # set USE VIDEO LIST
    USE_VIDEO_LIST = video_list.copy()

    patient_video_dict = {}

    # make patients video dict
    for patient in patient_list :
        pateint_parser = patient + '_' # R_1 => R_1_ | R_10 => R_10_
        patient_video = return_elements_is_str_in_list(pateint_parser, USE_VIDEO_LIST)

        patient_video_dict[patient] = patient_video
    
    print('\n----- RESULTS [PATIENT CNT : {} ] ------\n'.format(len(patient_video_dict)))
    print('----- PATIENT : {} ------\n'.format(patient_list))
    for keys, value in patient_video_dict.items() : 
        print('{} | {}'.format(keys, value))

    return patient_video_dict

def print_patients_info_yaml(patients_info_yaml_path): 
    print('\n\n\n\t\t\t ### [PATINET INFO] ### \n')
    
    with open(patients_info_yaml_path, 'r') as f :
        load_patients = yaml.load(f, Loader=yaml.FullLoader)

    patients = load_patients['patients']
    patient_count = len(patients)

    print('PATIENT COUNT : {}'.format(patient_count))
    print('PATIENT LIST')
    for idx in range(patient_count) : 
        print('-', patients[idx]['patient_no'])
    print('\n\n\t\t\t ### ### ### ### ### ### \n\n')
    
    for idx in range(patient_count) : 
        patient = patients[idx]

        print('PATIENT_NO : \t\t{}'.format(patient['patient_no']))
        print('PATIENT_VIDEO : \t{}\n'.format(patient['patient_video']))
        
        for video_path_info in patient['path_info'] :
            print('VIDEO_NAME : \t\t{}'.format(video_path_info['video_name']))
            print('VIDEO_PATH : \t\t{}'.format(video_path_info['video_path']))
            print('ANNOTATION_PATH : \t{}'.format(video_path_info['annotation_path']))
            print('DB_PATH : \t\t{}'.format(video_path_info['DB_path']))
            print('\n', '-----'*10, '\n')

        print('\n', '=== === === === ==='*5, '\n')

    '''
    @ EXAMPLE
    with open('./patinets_info.yaml', 'r') as f :
        load_patient = yaml.load(f, Loader=yaml.FullLoader)
    
    patients = load_patient

    print(len(patients))
    print(len(patients['patients']))
    print(patients['patients'][0]['patient_no'])
    print(patients['patients'][0]['patient_video'])
    print(patients['patients'][0]['path_info'][0]['video_name'])
    print(patients['patients'][0]['path_info'][0]['video_path'])
    print(patients['patients'][0]['path_info'][0]['annotation_path'])
    print(patients['patients'][0]['path_info'][0]['DB_path'])

    print('')
    print(patients['patients'][0]['path_info'][1]['video_name'])
    print(patients['patients'][0]['path_info'][1]['video_path'])
    print(patients['patients'][0]['path_info'][1]['annotation_path'])
    print(patients['patients'][0]['path_info'][1]['DB_path'])
    '''




# Pateint
    '''
    patients: 
        - patient_no: 'R_210'
          patient_video: ['R_210_ch1_03', 'R_210_ch2_04']
          path_info:
              - video_name : 'R_210_ch1_03'
                video_path : '/VIDEO_PATH'
                annotation_path : '/ANNOTATION_PATH'
                DB_path : '/DB_PATH'
              - video_name : 'R_210_ch2_04'
                video_path : 'VIDEO_PATH'
                annotation_path : '/ANNOTATION_PATH'
                DB_path : '/DB_PATH'
    '''
def make_patients_aggregate_info(patient_list, VIDEO_PATH_SHEET, ANNOTATION_PATH_SHEET, DB_PATH_SHEET, save_path): 


    # 1. SET patient video
    patient_video_dict = parsing_patient_video(patient_list, OOB_robot_list + OOB_lapa_list) # parsing pateint video

    # 2. patient video sorting
    for patient, video_name_list in patient_video_dict.items() : 
        patient_video_dict[patient] = pateint_video_sort(video_name_list)
    
    # print patinet video
    print('\n----- SORTED ------\n')
    for patient, video_name_list in patient_video_dict.items() : 
        print(patient, video_name_list)
    
    # 3. aggregtation to yaml
    patients = {'patients': []}

    # add 'patinet' obj
    for patient, video_name_list in patient_video_dict.items() : 

        path_info = []

        # add 'info' obj
        for video_name in video_name_list: 
            path_info.append({
                'video_name': video_name,
                'video_path': VIDEO_PATH_SHEET[video_name],
                'annotation_path': ANNOTATION_PATH_SHEET[video_name],
                'DB_path': DB_PATH_SHEET[video_name]
            })

        patients['patients'].append({
            'patient_no': patient,
            'patient_video': video_name_list,
            'path_info': path_info
        })
        

    # 4. serialization from python object to YAML stream and save
    with open(save_path, 'w') as f :
        yaml.dump(patients, f)
    


# sort [R_999_ch1_03, R_999_ch2_01, R_999_ch2_02] => [R_999_ch2_01, R_999_ch2_02, R_999_ch1_03]
# sort [01_G_01_R_999_ch1_03.mp4, 01_G_01_R_999_ch2_01.mp4, 01_G_01_R_999_ch2_02.mp4] => [01_G_01_R_999_ch2_01.mp4, 01_G_01_R_999_ch2_02.mp4, 01_G_01_R_999_ch1_03.mp4]
def pateint_video_sort(patient_video_list) :
    
    sorted_list = natsort.natsorted(patient_video_list, key=lambda x : os.path.splitext(x)[0].split('_')[-1], alg=natsort.ns.INT)
    
    return sorted_list



def annotation_parser(annotation_path, convert_fps=30): # default fps = 30 (because, only robot has csv annotation) 
    _, ext = os.path.splitext(annotation_path)

    assert ext in ['.csv', '.json'], 'CANT PARSING, SUPPOERT ANNOTATION FORMAT [.csv, .json] | TARGET FORMAT : {}'.format(ext)

    annotation_info = []

    if ext == '.csv' : # csv
        anno_df = pd.read_csv(annotation_path)
        anno_df = anno_df.dropna(axis=0) # 결측행 제거

        # time -> frame idx
        for i in range(len(anno_df)) :
            t_start = anno_df.iloc[i]['start'] # time
            t_end = anno_df.iloc[i]['end'] # time
            
            annotation_info.append([time_to_idx(t_start, convert_fps), time_to_idx(t_end, convert_fps)]) # annotation_info = [[start, end], [start, end]..]
                    
    elif ext == '.json' : # json
        with open(annotation_path) as json_file :
                json_data = json.load(json_file)

        # annotation frame
        for anno_data in json_data['annotations'] :
            t_start = anno_data['start'] # frame
            t_end = anno_data['end'] # frame

            annotation_info.append([t_start, t_end]) # annotation_info = [[start, end], [start, end]..]

    # when annotation contents nothing, return []
    return annotation_info



def patients_yaml_to_test_info_dict(patients_yaml_path) : # paring video from annotation info
    """
    Generate 'info_dict' (Incomplete form of 'info_dict') 

    Args:
        patients_yaml_path : patinets_yaml file (use def make_patients_aggregate_info())

    Returns:
        info_dict: 
            info_dict = {
                'video': [],
                'anno': [],
                'DB': [],
                'inference_assets' : []
            }
    """

    print('\n\n\n\t\t\t ### STARTING DEF [patients_yaml_to_test_info_dict] ### \n\n')

    with open(patients_yaml_path, 'r') as f : # in yaml, list sequence is preserved, key:value sequence is not preserved
        load_patients = yaml.load(f, Loader=yaml.FullLoader)
    
    patients = load_patients['patients']
    patients_count = len(patients)

    info_dict = {
        'video': [],
        'anno': [],
        'DB': [],
        'inference_assets' : []
    }

    for idx in range(patients_count): 
        patient = patients[idx]
        patient_no = patient['patient_no']
        patient_video = patient['patient_video']

        # it will be append to info_dict
        target_video_list = []
        target_anno_list = []
        target_inference_assets_list = []
        target_DB_list = []

        
        for video_path_info in patient['path_info']:
            video_name = video_path_info['video_name']
            video_path = video_path_info['video_path']
            annotation_path = video_path_info['annotation_path']
            DB_path = video_path_info['DB_path']

            print(video_name)

            # init
            temp_inference_assets_list = []
            # it will be append to temp_anno_list
            target_idx_list = []
            
            if annotation_path != '': # EVENT
                target_idx_list = annotation_parser(annotation_path)
            
            # save gettering info
            target_video_list.append(video_path) # [video1_1, video_1_2, ...]
            target_anno_list.append(target_idx_list) # [temp_idx_list_1_1, temp_idx_list_1_2, ... ]
            target_inference_assets_list.append(temp_inference_assets_list) # [video1_1_0, video_1_1_1, video_1_1_2, ..]
            target_DB_list.append(DB_path) # [DB1_1, DB1_2, ...]

        # info_dict['video'], info_dict['anno'] info_dict['DB'] length is same as valset
        info_dict['video'].append(target_video_list) # [[video1_1, video1_2], [video2_1, video_2_2], ...]
        info_dict['anno'].append(target_anno_list) # [[temp_idx_list_1_1, temp_idx_list_1_2], [temp_idx_list_2_1, temp_idx_list_2_2,], ...]
        info_dict['inference_assets'].append(target_inference_assets_list) # [[[video1_1_0, video1_1_1, video1_1_2,..], [video1_2_0, ...]], ... ]
        info_dict['DB'].append(target_DB_list) # [[DB1_1, DB1_2], [DB2_1,DB2_2], ...] # if no DB info, just append[['', ''], ['', ''], ...]

        print('\n\n')
    
    return info_dict


def save_dict_to_yaml(save_dict, save_path): # dictonary, ~.yaml
    save_dir, _ = os.path.split(save_path)
    try :
        if not os.path.exists(save_dir) :
            os.makedirs(save_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + save_dir)
    
    with open(save_path, 'w') as f :
        yaml.dump(save_dict, f)
    
def load_yaml_to_dict(yaml_file_path): # ~.yaml
    load_dict = {}

    with open(yaml_file_path, 'r') as f :
        load_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    return load_dict

    

# OOB_robot_40과 같은 GLOBAL 변수는 Main에서만 사용 (각 함수에서 내에서 사용지양, local 변수로 사용지향)
def make_data_sheet(save_dir):

    # DATA PATH SHEET
    ROBOT_VIDEO_PATH_SHEET = {}
    LAPA_VIDEO_PATH_SHEET = {}

    ROBOT_ANNOTATION_PATH_SHEET = {}
    LAPA_ANNOTATION_PATH_SHEET = {}

    ROBOT_DB_PATH_SHEET = {}
    LAPA_DB_PATH_SHEET = {}
    
    
    # 1. SET VIDEO PATH
    ROBOT_DATASET_1_VIDEO_ROOT_PATH = '/data1/HuToM/Video_Robot_cordname' # Dataset 1 - ROBOT
    ROBOT_DATASET_2_VIDEO_ROOT_PATH = '/data2/Video/Robot/Dataset2_60case' # Dataset 2 - ROBOT
    ROBOT_VIDEO_PATH_SHEET = {**get_video_path_for_robot(ROBOT_DATASET_1_VIDEO_ROOT_PATH, OOB_robot_40), **get_video_path_for_robot(ROBOT_DATASET_2_VIDEO_ROOT_PATH, OOB_robot_60)}

    LAPA_DATASET_1_VIDEO_ROOT_PATH = '/data2/Public/IDC_21.06.25/Dataset1' # Dataset 1 - LAPA
    LAPA_DATASET_2_VIDEO_ROOT_PATH = '/data2/Public/IDC_21.06.25/Dataset2' # Dataset 2 - LAPA
    LAPA_VIDEO_PATH_SHEET = {**get_video_path(LAPA_DATASET_1_VIDEO_ROOT_PATH, OOB_lapa_40), **get_video_path(LAPA_DATASET_2_VIDEO_ROOT_PATH, OOB_lapa_60)}

    # 2. SET ANNOTATION PATH
    ANNOTATION_V1_ROOT_PATH = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1'
    ANNOTATION_V2_ROOT_PATH = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2'
    
    ROBOT_ANNOTATION_PATH_SHEET = get_anno_path(ANNOTATION_V1_ROOT_PATH, OOB_robot_list) # V1 - ROBOT
    LAPA_ANNOTATION_PATH_SHEET = get_anno_path(ANNOTATION_V1_ROOT_PATH, OOB_lapa_list) # V1 - LAPA
    
    # 3. SET DB PATH
    ROBOT_DB_ROOT_PATH = '/data2/Public/OOB_Recog/img_db/ROBOT'
    LAPA_DB_ROOT_PATH = '/data2/Public/OOB_Recog/img_db/LAPA'
    
    ROBOT_DB_PATH_SHEET = get_DB_path(ROBOT_DB_ROOT_PATH, OOB_robot_list) # ROBOT
    LAPA_DB_PATH_SHEET = get_DB_path(LAPA_DB_ROOT_PATH, OOB_lapa_list) # LAPA

    # 4. AGGREGATE
    VIDEO_PATH_SHEET = {**ROBOT_VIDEO_PATH_SHEET, **LAPA_VIDEO_PATH_SHEET}
    ANNOTATION_PATH_SHEET = {**ROBOT_ANNOTATION_PATH_SHEET, **LAPA_ANNOTATION_PATH_SHEET} # V1
    DB_PATH_SHEET = {**ROBOT_DB_PATH_SHEET, **LAPA_DB_PATH_SHEET}

    # 5. SAVE SHEET
    VIDEO_PATH_SHEET_SAVE_PATH = os.path.join(save_dir, 'VIDEO_PATH_SHEET.yaml')
    ANNOTATION_SHHET_SAVE_PATH = os.path.join(save_dir, 'ANNOTATION_PATH_SHEET.yaml')
    DB_SHEET_SAVE_PATH = os.path.join(save_dir, 'DB_PATH_SHEET.yaml')

    save_dict_to_yaml(VIDEO_PATH_SHEET, VIDEO_PATH_SHEET_SAVE_PATH)
    save_dict_to_yaml(ANNOTATION_PATH_SHEET, ANNOTATION_SHHET_SAVE_PATH)
    save_dict_to_yaml(DB_PATH_SHEET, DB_SHEET_SAVE_PATH)

def load_data_sheet(data_sheet_dir):
    # 1. set load path from data_sheet_dir
    VIDEO_PATH_SHEET_yaml_path = os.path.join(data_sheet_dir, 'VIDEO_PATH_SHEET.yaml')
    ANNOTATION_PATH_SHEET_yaml_path = os.path.join(data_sheet_dir, 'ANNOTATION_PATH_SHEET.yaml')
    DB_PATH_SHEET_yaml_path = os.path.join(data_sheet_dir, 'DB_PATH_SHEET.yaml')
    
    # 2. load from yaml to dict
    VIDEO_PATH_SHEET = load_yaml_to_dict(VIDEO_PATH_SHEET_yaml_path)
    ANNOTATION_PATH_SHEET = load_yaml_to_dict(ANNOTATION_PATH_SHEET_yaml_path)
    DB_PATH_SHEET = load_yaml_to_dict(DB_PATH_SHEET_yaml_path)

    return VIDEO_PATH_SHEET, ANNOTATION_PATH_SHEET, DB_PATH_SHEET





def main():
    # make DATA SHEET
    make_data_sheet('./DATA_SHEET')

    # load DATA SHEET
    VIDEO_PATH_SHEET, ANNOTATION_PATH_SHEET, DB_PATH_SHEET = load_data_sheet('./DATA_SHEET')

    patient_list = ['R_210', 'R_424', 'R_391', 'L_676', 'E_999']

    # make patinets aggregation file
    make_patients_aggregate_info(patient_list, VIDEO_PATH_SHEET, ANNOTATION_PATH_SHEET, DB_PATH_SHEET, './patients_info.yaml')

    # print patinets aggregation file
    print_patients_info_yaml('./patients_info.yaml')
    
    # convert patients aggregation file to info_dict
    info_dict = patients_yaml_to_test_info_dict('./patients_info.yaml')

    print(info_dict)




if __name__ == "__main__":
	main()