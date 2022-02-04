'''
patient name, video name, fps, TotalFrame, frame cnt, RS, NRS_wo_ssim, NRS_w_ssim, NRS_w_ssim_duplicate, time threshold, ssim score threshold,
R_1, 
'''

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


def get_cnt(assets_df):
    rs_non_duplicate = len(assets_df[assets_df['class']==0])
    rs_duplicate = len(assets_df[assets_df['class']==1])
    nrs_non_duplicate = len(assets_df[assets_df['class']==2])
    nrs_duplicate = len(assets_df[assets_df['class']==3])

    return rs_non_duplicate, rs_duplicate, nrs_non_duplicate, nrs_duplicate

def report_per_video(assets_df, target_fps, time_th, ssim_score_th, save_path):
    field_names = ['patient_name', 'target_fps', 'Total', 'RS-non_duplicate', 'RS-duplicate', 'NRS-non_duplicate', 'NRS-duplicate', 'total-norm', 'rs-norm', 'nrs-norm', 'time_th', 'ssim_score_th']

    assets_df['video_name'] = assets_df['frame_path'].str.split('/').str[-2] # column : video name
    video_list = list(assets_df['video_name'].unique()) # [04_GS4_99_L_1_01, # 04_GS4_99_L_1_02]

    patient_name_list = []
    video_name_list = []
    target_fps_list = []
    Total_list = []

    RS_non_duplicate_list = []
    RS_duplicate_list = []
    NRS_non_duplicate_list = []
    NRS_duplicate_list = []
    
    total_norm_list = []
    RS_norm_list = []
    NRS_norm_list = []
    
    time_th_list = []
    ssim_score_th_list = []

    # report per video
    group_df = assets_df.groupby(assets_df.video_name)
    for video in video_list: # 04_GS4_99_L_1_01
        video_df = group_df.get_group(video)

        rs_non_duplicate, rs_duplicate, nrs_non_duplicate, nrs_duplicate = get_cnt(video_df)

        try:
            total_norm = (rs_duplicate+nrs_duplicate)/len(assets_df)
        except:
            total_norm = -1
        
        try:
            rs_norm = rs_duplicate/(rs_non_duplicate+rs_duplicate)
        except:
            rs_norm = -1

        try:
            nrs_norm = nrs_duplicate/(nrs_non_duplicate+nrs_duplicate)
        except:
            nrs_norm = -1


        patient_name_list.append('_'.join(video.split('_')[:5]))
        video_name_list.append(video)
        target_fps_list.append(target_fps)
        Total_list.append(len(video_df))

        RS_non_duplicate_list.append(rs_non_duplicate)
        RS_duplicate_list.append(rs_duplicate)
        NRS_non_duplicate_list.append(nrs_non_duplicate)
        NRS_duplicate_list.append(nrs_duplicate)
        
        total_norm_list.append(total_norm)
        RS_norm_list.append(rs_norm)
        NRS_norm_list.append(nrs_norm)
        
        time_th_list.append(time_th)
        ssim_score_th_list.append(ssim_score_th)


    # patient name, video name, fps, TotalFrame, frame cnt, RS, NRS_wo_ssim, NRS_w_ssim, NRS_w_ssim_duplicate, time threshold, ssim score threshold,
    report_per_video = {
        'patient_name': patient_name_list,
        'video_name': video_name_list,
        'target_fps': target_fps_list,
        'Total': Total_list,
        
        'RS-non_duplicate': RS_non_duplicate_list,
        'RS-duplicate': RS_duplicate_list,
        'NRS-non_duplicate': NRS_non_duplicate_list,
        'NRS-duplicate': NRS_duplicate_list,
        
        'total-norm':total_norm_list,
        'rs-norm': RS_norm_list,
        'nrs-norm': NRS_norm_list,
        
        'time_th': time_th_list,
        'ssim_score_th': ssim_score_th_list,
    }

    report_per_video_df = pd.DataFrame(report_per_video)

    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'REPORT-{}-video.csv'.format('_'.join(video.split('_')[:5])+'.csv'))

    
    if not os.path.isfile(save_path):
       report_per_video_df.to_csv(save_path, mode='w', sep=',')

    else:
        report_per_video_df.to_csv(save_path, mode='a', header=not os.path.exists(save_path))

    

def report_per_patient(assets_df, patient_name, target_fps, time_th, ssim_score_th, save_path, VIDEO_FPS):
    rs_non_duplicate, rs_duplicate, nrs_non_duplicate, nrs_duplicate = get_cnt(assets_df)
    
    field_names = ['patient_name', 'VIDEO_FPS', 'target_fps', 'Total', 'RS-non_duplicate', 'RS-duplicate', 'NRS-non_duplicate', 'NRS-duplicate', 'total-norm', 'rs-norm', 'nrs-norm', 'time_th', 'ssim_score_th']
    
    try:
        total_norm = (rs_duplicate+nrs_duplicate)/len(assets_df)
    except:
        total_norm = -1
    
    try:
        rs_norm = rs_duplicate/(rs_non_duplicate+rs_duplicate)
    except:
        rs_norm = -1

    try:
        nrs_norm = nrs_duplicate/(nrs_non_duplicate+nrs_duplicate)
    except:
        nrs_norm = -1

    # patient name, video name, fps, TotalFrame, frame cnt, RS, NRS_wo_ssim, NRS_w_ssim, NRS_w_ssim_duplicate, time threshold, ssim score threshold,
    report_per_patient = {
        'patient_name': patient_name,
        'VIDEO_FPS': VIDEO_FPS,
        'target_fps': target_fps,
        'Total': len(assets_df),

        'RS-non_duplicate': rs_non_duplicate,
        'RS-duplicate': rs_duplicate,
        'NRS-non_duplicate': nrs_non_duplicate,
        'NRS-duplicate': nrs_duplicate,
        
        'total-norm':total_norm,
        'rs-norm': rs_norm,
        'nrs-norm': nrs_norm,
        
        'time_th': time_th,
        'ssim_score_th': ssim_score_th,
    }

    report_per_patient_df = pd.DataFrame([report_per_patient])
    
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'REPORT-{}_{}FPS-patient.csv'.format(save_path.split('/')[-1], target_fps))

    if not os.path.isfile(save_path):
        report_per_patient_df.to_csv(save_path, mode='w', sep=',')

    else:
        report_per_patient_df.to_csv(save_path, mode='a', header=not os.path.exists(save_path))




if __name__ == '__main__':
    with open('./assets/L_1/L_1-1_fps.pickle', 'rb') as f:
        data1 = pickle.load(f)

    with open('./assets/L_1/L_1-5_fps.pickle', 'rb') as f:
        data2 = pickle.load(f)

    report_per_video_save_path = 'ssim_result/gangbuksamsung_127case/04_GS4_99_L_1'
    report_per_video(assets_df=data2, target_fps='5', time_th=0, ssim_score_th=0.95, save_path=report_per_video_save_path)

    report_per_patient_save_path = 'ssim_result/gangbuksamsung_127case'
    report_per_patient(assets_df=data2, patient_name='04_GS4_99_L_1', target_fps='5', time_th=0, ssim_score_th=0.95, save_path=report_per_patient_save_path)

    report_per_video_save_path = 'ssim_result/gangbuksamsung_127case/04_GS4_99_L_1'
    report_per_video(assets_df=data1, target_fps='1', time_th=0, ssim_score_th=0.95, save_path=report_per_video_save_path)

    report_per_patient_save_path = 'ssim_result/gangbuksamsung_127case'
    report_per_patient(assets_df=data1, patient_name='04_GS4_99_L_1', target_fps='1', time_th=0, ssim_score_th=0.95, save_path=report_per_patient_save_path)

