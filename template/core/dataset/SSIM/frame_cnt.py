
import os
import glob
import natsort

import pandas as pd
import json

def frame_cnt():
    # base_path = '/raid/img_db/VIHUB/gangbuksamsung_127case'

    # base_path = '/raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_A9'
    # base_path = '/raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_B4'
    # base_path = '/raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_B5'

    # base_path = '/raid/img_db/VIHUB/severance_2nd/01_VIHUB1.2_A9'
    base_path = '/raid/img_db/VIHUB/severance_2nd/01_VIHUB1.2_B4'

    patient_path_list = glob.glob(os.path.join(base_path, '*'))
    patient_path_list = natsort.natsorted(patient_path_list)

    patient_save_list = []
    frame_save_list = []

    for patient_path in patient_path_list:
        frame_list = glob.glob(os.path.join(patient_path, '*', '*.jpg'))    
        
        patient_save_list.append(patient_path.split('/')[-1])
        frame_save_list.append(len(frame_list))

    data = {
        'paient': patient_save_list,
        'frame_cnt': frame_save_list
    }

    df = pd.DataFrame(data)

    df.to_csv('./s2-{}-frame_cnt.csv'.format(base_path.split('/')[-1]))

def frame_cnt_per_video():
    # base_path = '/raid/img_db/VIHUB/gangbuksamsung_127case'

    # base_path = '/raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_A9'
    # base_path = '/raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_B4'
    base_path = '/raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_B5'

    # base_path = '/raid/img_db/VIHUB/severance_2nd/01_VIHUB1.2_A9'
    # base_path = '/raid/img_db/VIHUB/severance_2nd/01_VIHUB1.2_B4'

    video_path_list = glob.glob(os.path.join(base_path, '*', '*'))
    video_path_list = natsort.natsorted(video_path_list)

    video_save_list = []
    frame_save_list = []

    for video_path in video_path_list:
        frame_list = glob.glob(os.path.join(video_path, '*.jpg'))    
        
        video_save_list.append(video_path.split('/')[-1])
        frame_save_list.append(len(frame_list))

    data = {
        'video': video_save_list,
        'frame_cnt': frame_save_list
    }

    df = pd.DataFrame(data)

    df.to_csv('./s1-{}-frame_cnt_per_video.csv'.format(base_path.split('/')[-1]))

    
def video_cnt():
    # video_path = '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case'
    
    # video_path = '/data3/DATA/IMPORT/211220/12_14/severance_1st/01_VIHUB1.2_A9'
    # video_path = '/data3/DATA/IMPORT/211220/12_14/severance_1st/01_VIHUB1.2_B4'
    # video_path = '/data3/DATA/IMPORT/211220/12_14/severance_1st/01_VIHUB1.2_B5'
    
    # video_path = '/data3/DATA/IMPORT/211220/12_14/severance_2nd/01_VIHUB1.2_A9'
    video_path = '/data3/DATA/IMPORT/211220/12_14/severance_2nd/01_VIHUB1.2_B4'

    video_list = glob.glob(os.path.join(video_path, '*', '*'))

    data = {
        'video': video_list
    }

    df = pd.DataFrame(data)
    df.to_csv('./s2-{}-video.csv'.format(video_path.split('/')[-1]))
    

def read_totalFrame():
    json_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_2nd/NRS'

    json_list = glob.glob(os.path.join(json_path, '*'))
    json_list = natsort.natsorted(json_list)

    video_save = []
    json_save_data = []
    patient_list = []

    for json_f in json_list:
        with open(json_f, 'r') as f:
            json_data = json.load(f)

        total_anno = json_data['totalFrame']
        video_name = json_data['name']

        patient_list.append('_'.join(video_name.split('_')[:-1]))

        video_save.append(video_name)
        json_save_data.append(total_anno)


    d = {
        'patient': patient_list,
        'video': video_save,
        'totalFrame': json_save_data
    }

    df = pd.DataFrame(d)

    df.to_csv('./{}-frame_per_video.csv'.format(json_path.split('/')[-2]))


def read_fps():
    # json_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS'
    # json_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_1st/NRS'
    json_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_2nd/NRS'

    json_list = glob.glob(os.path.join(json_path, '*'))
    json_list = natsort.natsorted(json_list)

    video_save = []
    json_save_data = []
    patient_list = []

    for json_f in json_list:
        with open(json_f, 'r') as f:
            json_data = json.load(f)

        frame_rate = json_data['frameRate']
        video_name = json_data['name']

        video_save.append(video_name)
        json_save_data.append(frame_rate)


    d = {
        'video': video_save,
        'totalFrame': json_save_data
    }

    df = pd.DataFrame(d)

    df.to_csv('./0127-{}-frame_rate.csv'.format(json_path.split('/')[-2]))


read_fps()
