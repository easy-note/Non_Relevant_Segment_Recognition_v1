
import os
import glob

import csv
import json


from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd

import sys
import cv2

import natsort

## Annotation path
# annotation_v1_base_path = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1'
annotation_v2_base_path = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2'

## img path
# img_base_path = '/data2/Public/OOB_Recog/img_db'
img_base_path = '/raid/img_db'

## oob_assets save path
# oob_assets_v1_robot_save_path = '/data2/Public/OOB_Recog/oob_assets/V1/ROBOT'
# oob_assets_v1_lapa_save_path = '/data2/Public/OOB_Recog/oob_assets/V1/LAPA'
# oob_assets_v2_robot_save_path = '/data2/Public/OOB_Recog/oob_assets/V2/ROBOT'
# oob_assets_v2_lapa_save_path = '/data2/Public/OOB_Recog/oob_assets/V2/LAPA'
oob_assets_v2_robot_save_path = '/raid/img_db/oob_assets/V2/ROBOT'
oob_assets_v2_lapa_save_path = '/raid/img_db/oob_assets/V2/LAPA'


def save_list_to_csv(save_filepath, list, mode):
    with open(save_filepath, mode) as f:
        writer = csv.writer(f)
        for row in list:
            writer.writerow(row)

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory {}'.format(directory))


def parsing_oob_list(json_file):
    with open(json_file) as json_file:
        time_list = []
        json_data = json.load(json_file)
        json_anno = json_data["annotations"]
        
        for list in json_anno:
            temp_list = []
            temp_list.append(int(list.get("start"))) # float -> int
            temp_list.append(int(list.get("end"))) # float -> int
            time_list.append(temp_list)

        oob_list = []
        for time in time_list:
            for i in range(time[0], time[1] + 1):
                oob_list.append('{:010d}'.format(i))

    return oob_list

def frame_length_parity_check(json_file):
    with open(json_file) as json_file:
        json_data = json.load(json_file)
        json_total_frame = json_data["totalFrame"]
    
    return json_total_frame

def save_log(save_path, log_txt):
    with open(save_path, 'a') as f:
        f.write(log_txt+'\n')
    
# annotation_version_base_path, Device (ROBOT, LAPA)
def make_oob_csv(anno_base_path, device):
    total_annotation_list = glob.glob(anno_base_path + '/*')

    if device.lower() == 'robot':
        anno_list = [i for i in total_annotation_list if '_R_' in i]
        anno_list = natsort.natsorted(anno_list)
        print('anno_list : {}\n\n'.format(anno_list))
        # target_img_base_path = '/data2/Public/OOB_Recog/img_db/ROBOT' # 현재 IDC_NAS_path -> IDC 서버로 img_db 모두 copy 이후는, IDC 서버 경로 잡을 예정. 
        target_img_base_path = os.path.join(img_base_path, 'ROBOT')

    elif device.lower() == 'lapa':
        anno_list = [i for i in total_annotation_list if '_L_' in i]
        anno_list = natsort.natsorted(anno_list)
        print('anno_list : {}\n\n'.format(anno_list))
        # target_img_base_path = '/data2/Public/OOB_Recog/img_db/LAPA' # 현재 IDC_NAS_path -> IDC 서버로 img_db 모두 copy 이후는, IDC 서버 경로 잡을 예정. 
        target_img_base_path = os.path.join(img_base_path, 'LAPA')

    error_list = []

    outbody_list = []
    inbody_list = []

    for anno_file in anno_list:
        create_folder('/raid/img_db/oob_assets/V2/ROBOT')
        save_log('/raid/img_db/oob_assets/V2/ROBOT/train_assets_log.txt', anno_file)

        try:
            if 'json' in anno_file:
                print('Processing in ====> {}\n'.format(anno_file))
                oob_list = parsing_oob_list(anno_file)
                print('oob_list : {}\n\n'.format(oob_list))
            else:
                continue

        except:
            print('====' * 5)
            print('ERROR: cannot parsing oob_list ====> {}'.format(anno_file))
            print('====' * 5)
            error_list.append([anno_file])

        patient_folder_name = '_'.join(anno_file.split('/')[-1].split('_')[3:5]) # R_94
        video_folder_name = '_'.join(anno_file.split('/')[-1].split('_')[:7]) # 01_G_01_R_94_ch1_03

        target_img_path = os.path.join(target_img_base_path, patient_folder_name, video_folder_name)
        target_img_list = glob.glob(os.path.join(target_img_path, '*.jpg'))
        
        target_img_list_length = len(target_img_list)

        
        ## annotation total frame, target_img_list length parity check.
        if frame_length_parity_check(anno_file) >= target_img_list_length:
            print('target_img_path : {}\n'.format(target_img_path))

            for target_img in target_img_list:
                target_img_idx = target_img.split('-')[1][:-4]
                
                if int(target_img_idx) % 30 == 0: # train step : 30
                    if target_img_idx in oob_list:
                        outbody_list.append([target_img, 1])

                    else:
                        inbody_list.append([target_img, 0])

        ## if annotation total frame < target img list
        else : 
            target_img_list = natsort.natsorted(target_img_list)
            print('ORIGIN FRAME LENGTH ====> {}'.format(target_img_list_length))

            target_img_list  = target_img_list[:frame_length_parity_check(anno_file)]
            print('CONVERT FRAME LENGTH ====> {}'.format(len(target_img_list)))
            print('\t====>', frame_length_parity_check(anno_file), len(target_img_list))

            for target_img in target_img_list:
                target_img_idx = target_img.split('-')[1][:-4]
                
                if int(target_img_idx) % 30 == 0: # train step : 30
                    if target_img_idx in oob_list:
                        outbody_list.append([target_img, 1])

                    else:
                        inbody_list.append([target_img, 0])
            
    inbody_list = natsort.natsorted(inbody_list)
    outbody_list = natsort.natsorted(outbody_list)

    # ## save
    # if 'V1' in anno_base_path:
    #     if device.lower() == 'robot':
    #         output_save_path = oob_assets_v1_robot_save_path
    #     else:
    #         output_save_path = oob_assets_v1_lapa_save_path
    
    # elif 'V2' in anno_base_path:
    if device.lower() == 'robot':
        output_save_path = oob_assets_v2_robot_save_path
    else:
        output_save_path = oob_assets_v2_lapa_save_path

    create_folder(output_save_path)

    save_list_to_csv(os.path.join(output_save_path, 'oob_assets_outofbody.csv'), outbody_list, 'w')
    save_list_to_csv(os.path.join(output_save_path, 'oob_assets_inbody.csv'), inbody_list, 'w')

    # if error
    if error_list:
        print(error_list)
        save_list_to_csv(os.path.join(output_save_path, 'ERROR_img.csv'), error_list, 'w')


if __name__ == '__main__':
    make_oob_csv(annotation_v2_base_path, 'robot')
    # make_oob_csv(annotation_v2_base_path, 'lapa')

