import os
import glob
import cv2
import random
import numpy as np
import pandas as pd
from pandas import DataFrame as df

import datetime
import time

from tqdm import tqdm

import json
import math

from decord import VideoReader
from decord import cpu, gpu

# cal vedio frame
def time_to_idx(time, fps):
    t_segment = time.split(':')
    idx = (int(t_segment[0]) * 3600 * fps) + (int(t_segment[1]) * 60 * fps) + (int(t_segment[2]) * fps) + int(t_segment[3]) # [h, m, s, frame] 

    return idx

def idx_to_time(idx, fps) :
    time_s = idx // fps
    frame = idx % fps

    converted_time = str(datetime.timedelta(seconds=time_s))
    converted_time = converted_time + ':' + str(frame)

    return converted_time

def save_log(log_txt, save_dir) :
    # first log file
    if not os.path.exists(save_dir) :
        print('=========> CREATE INIT LOG FILE ... | {}'.format(save_dir))
        with open(save_dir, 'w') as f :
            f.write(log_txt) 
    else :
        print('=========> SAVING LOG ... | {}'.format(save_dir))
        with open(save_dir, 'a') as f :
            f.write(log_txt)

# Annotation Sanity Check
def check_anno_sequence(anno_info:list): # annotation sequence에 이상없을 경우 = True, 이상 있을경우 = False

    if len(anno_info) == 1 :
        p_start, p_end = anno_info[0][0], anno_info[0][1]
        is_block_seq_ok = False

        if p_start < p_end : 
            is_block_seq_ok = True

        if not(is_block_seq_ok) :
            return False


    elif len(anno_info) >  1 :
        p_start, p_end = anno_info[0][0], anno_info[0][1]
        for start, end in anno_info[1:] :
            is_block_seq_ok = False
            is_total_seq_ok = False

            if start < end : 
                is_block_seq_ok = True

            if p_end < start : 
                is_total_seq_ok = True 

            if not(is_block_seq_ok) or not(is_total_seq_ok) :
                return False

            p_start, p_end = start, end

    return True

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


# data paring with csv
def gettering_information_for_oob(video_root_path, anno_root_path, video_set, mode) : # paring video from annotation info
    print('\n\n\n\t\t\t ### STARTING DEF [gettering_information_for_oob] ### \n\n')
    
    info_dict = {
        'video': [],
        'anno': [],
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
        all_video_path.remove(os.path.join(video_root_path, '01_G_01_L_423_xx0_01.MP4'))
        all_anno_path.remove(os.path.join(anno_root_path, '01_G_01_L_423_xx0_01_OOB_16.json'))

        # print(set(os.path.splitext(os.path.basename(path))[1] for path in glob.glob(video_root_path + '/*'))) # all file ext check
        # ext_video = [path for path in glob.glob(video_root_path + '/*') if os.path.splitext(os.path.basename(path))[1]!='.MP4'] # video ext file
        # for video in video_set :
        #     print(video)
        #     print([p for p in ext_video if p.find(video+'_')!=-1])

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

        # extract only xx0, ch1
        video_path_df = video_path_df[video_path_df['video_path'].str.contains('xx0|ch1')]

        # sort video_path_df
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

        # df -> list 
        video_path_list = list(pair_info['video_path'])
        anno_path_list = list(pair_info['anno_path'])
        
        print('\t ==== GETTERING INFO ====')
        print('\t VIDEO NO | ', video_no) 
        print(pair_info)
        print('\t ----------------- \n')
        print('\t video_path', video_path_list) # target videos path
        print('\t anno_path', anno_path_list) # target annotaion path
        print('\t ==== ==== ==== ====\n')

        # it will be append to info_dict
        target_video_list = []
        target_anno_list = []

      
        for target_video_dir, target_anno_dir in (zip(video_path_list, anno_path_list)) :

            # it will be append to temp_anno_list
            target_idx_list = []

            # only target_video_dir 
            if target_anno_dir != '' :

                if mode=='ROBOT' : # csv
                    anno_df = pd.read_csv(target_anno_dir)
                    anno_df = anno_df.dropna(axis=0) # 결측행 제거

                    # time -> frame idx
                    for i in range(len(anno_df)) :
                        t_start = anno_df.iloc[i]['start'] # time
                        t_end = anno_df.iloc[i]['end'] # time
                        
                        target_idx_list.append([time_to_idx(t_start, fps), time_to_idx(t_end, fps)]) # temp_idx_list = [[start, end], [start, end]..]
                    
                    print('-----'*3)
                    print(target_video_dir)
                    print(target_anno_dir)
                    print(anno_df)
                
                elif mode=='LAPA' : # json
                    with open(target_anno_dir) as json_file:
                        json_data = json.load(json_file)

                    # open video cap, only check for frame count
                    video = cv2.VideoCapture(target_video_dir)
                    video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_fps = video.get(cv2.CAP_PROP_FPS)
                    video.release()

                    # annotation frame
                    for anno_data in json_data['annotations'] :
                        t_start = anno_data['start'] # frame
                        t_end = anno_data['end'] # frame
                        
                        # truncate
                        target_idx_list.append([t_start, t_end]) # temp_idx_list = [[start, end], [start, end]..]

                    print('-----'*3)
                    print(target_video_dir)
                    print(target_anno_dir)
                    print('ANNO \t FPS {} | TOTAL {}'.format(json_data['frameRate'], json_data['totalFrame']))
                    print('VIDEO \t FPS {} | TOTAL {}'.format(video_fps, video_len))
                    print(json_data['annotations'])
                    
            else :
                print('-----'*3)
                print(target_video_dir)
                print(target_anno_dir)
                print(target_idx_list)

            # save gettering info
            target_video_list.append(target_video_dir) # [video1_1, video_1_2, ...]
            target_anno_list.append(target_idx_list) # [temp_idx_list_1_1, temp_idx_list_1_2, ... ]

        # info_dict['video'], info_dict['anno'] length is same as valset
        info_dict['video'].append(target_video_list) # [[video1_1, video1_2], [video2_1, video_2_2], ...]
        info_dict['anno'].append(target_anno_list) # [[temp_idx_list_1_1, temp_idx_list_1_2], [temp_idx_list_2_1, temp_idx_list_2_2,], ...]
        
        print('\n\n')
        print(info_dict)
    
    return info_dict

# 불러온 annotation file 정보가 정확한지 체크 및 수정
def sanity_check_info_dict(info_dict) :
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
        
            # open VideoReader
            '''
            with open(video_path, 'rb') as f :
                # open VideoReader
                video = VideoReader(f, ctx=cpu(0))
            
            # total frame
            video_len = len(video) 

            del video
            '''

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

                # seq check
                if not(check_anno_sequence(anno_info)) : 
                    print('ANNTATION SEQ ERROR | video : {} | anno {}'.format(video_path, anno_info))
                    exit(1)
                
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

        



def gen_image_dataset_for_oob(video_root_dir, anno_root_dir, save_root_dir, capture_step, mode):
    # start time stamp
    startTime = time.time()
    s_tm = time.localtime(startTime)

    # create save folder
    oob_save_dir = os.path.join(save_root_dir, 'OutBody')
    ib_save_dir = os.path.join(save_root_dir, 'InBody')

    try :
        if not os.path.exists(save_root_dir) :
            os.makedirs(save_root_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + save_root_dir)

    try :
        if not os.path.exists(oob_save_dir) :
            os.makedirs(oob_save_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + oob_save_dir)

    try :
        if not os.path.exists(ib_save_dir) :
            os.makedirs(ib_save_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + ib_save_dir)

    log_txt='\n\n=============== \t\t START TIME \t\t ============= \n\n'
    log_txt+='STARTED AT : \t' + time.strftime('%Y-%m-%d %I:%M:%S %p \n', s_tm)
    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log

    log_txt='\n\n=============== \t\t ARGS INFO \t\t ============= \n\n'
    log_txt+='VIDEO ROOT DIR : \t {}\n'.format(video_root_dir)
    log_txt+='ANNOTATION ROOT DIR : \t {}\n'.format(anno_root_dir)
    log_txt+='SAVE ROOT DIR : \t {}\n'.format(save_root_dir)
    log_txt+='CAPTURE STEP : \t {}\n\n\n'.format(capture_step)
    
    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log

    robot_video_set = ['R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_10', 'R_13', 'R_14', 'R_15', 'R_17', 'R_18', 
                'R_19', 'R_22', 'R_48', 'R_56', 'R_74', 'R_76', 'R_84', 'R_94', 'R_100', 'R_116', 'R_117', 'R_201', 'R_202', 'R_203', 
                'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_301', 'R_302', 'R_303', 'R_304', 'R_305', 'R_313']
    
    lapa_video_set = ['L_301', 'L_303', 'L_305', 'L_309', 'L_317', 'L_325', 'L_326', 'L_340', 'L_346', 'L_349', 'L_412', 'L_421', 'L_423', 'L_442',
                'L_443', 'L_450', 'L_458', 'L_465', 'L_491', 'L_493', 'L_496', 'L_507', 'L_522', 'L_534', 'L_535', 'L_550',
                'L_553', 'L_586', 'L_595', 'L_605', 'L_607', 'L_625', 'L_631', 'L_647', 'L_654', 'L_659', 'L_660', 'L_661', 'L_669', 'L_676']

    # paring video from annotation info
    ### mode = ['ROBOT', 'LAPA']
    ### video_set = robot_video_set, lapa_video_set
    if mode=='ROBOT' : 
        info_dict = gettering_information_for_oob(video_root_dir, anno_root_dir, robot_video_set, mode='ROBOT')
    elif mode=='LAPA' :
        info_dict = gettering_information_for_oob(video_root_dir, anno_root_dir, lapa_video_set, mode='LAPA')
    else :
        assert False, 'ONLY SUPPORT MODE [ROBOT, LAPA] | Input mode : {}'.format(mode) 
    
    ### check info_dict and redifined
    info_dict = sanity_check_info_dict(info_dict)

    exit(0)

    total_videoset_cnt = len(info_dict['video']) # total number of video set
    print('{} VIDEO SET WILL BE CAPUERTRED ... '.format(total_videoset_cnt))

    Processed_OOB_cnt = 0
    Processed_IB_cnt = 0
    NON_READ_CNT = 0
    NON_WRITE_CNT = 0

    # loop from total_videoset_cnt
    for i, (video_path_list, anno_info_list) in enumerate(zip(info_dict['video'], info_dict['anno']), 1): 
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice = os.path.splitext(video_path_list[0])[0].split('_')
        videoset_name = '{}_{}'.format(op_method, patient_idx)

        print('COUNT OF VIDEO SET | {} / {} \t\t ======>  VIDEO SET | {}'.format(i, total_videoset_cnt, videoset_name))
        print('NUMBER OF VIDEO : {} | NUMBER OF ANNOTATION INFO : {}'.format(len(video_path_list), len(anno_info_list)))
        print('IMAGE SAVED AT \t\t\t ======>  {}'.format(save_root_dir))
        print('\n')

        for video_path, anno_info in zip(video_path_list, anno_info_list) :

            video_name = os.path.splitext(os.path.basename(video_path))[0] # only video name
            print(video_name)

            # open video cap
            video = cv2.VideoCapture(video_path)
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = video.get(cv2.CAP_PROP_FPS)

            print('\tTarget video : {} | Total Frame : {} | Video FPS : {} '.format(video_name, video_len, video_fps))
            print('\tAnnotation Info : {}'.format(anno_info))

            ### check idx -> time
            for start, end in anno_info :
                print([idx_to_time(start, 30), idx_to_time(end, 30)])
            
            print('')

            ####  make truth list ####
            IB_CLASS, OOB_CLASS = [0, 1]
            truth_list = np.zeros(video_len, dtype='uint8') if IB_CLASS == 0 else np.ones(video_len, dtype='uint8')
            for start, end in anno_info :
                truth_list[start:end+1] = OOB_CLASS # OOB Section

            truth_list = list(truth_list) # change to list

            truth_oob_count = truth_list.count(OOB_CLASS)

            print('IB_CLASS = {} | OOB_CLASS = {}'.format(IB_CLASS, OOB_CLASS))
            print('TRUTH IB FRAME COUNT : ', video_len - truth_oob_count)
            print('TRUTH OOB FRAME COUNT : ', truth_oob_count)
            ### ### ###

            temp_OOB_cnt = 0
            temp_IB_cnt = 0
            print('\n\n \t *** START *** \n\n')
            # captureing image and save 
            for frame_idx, truth in enumerate(tqdm(truth_list, desc='Capturing... \t ==> {}'.format(video_name))) :
                img = None
                save_file_name = None
                
                if frame_idx % capture_step != 0 :
                    continue

                is_set_ = video.set(1, frame_idx) # frame setting
                is_read_, img = video.read() # read frame
                
                # set/ read sanity check
                if not (is_set_ and is_read_) :
                    print('video_path : ', video_path)
                    print('video_name : ', video_name)
                    print('frame : ', frame_idx)
                    print('set : ', is_set_)
                    print('read : ', is_read_)
                    
                    READ_LOG='video_path : {}\t video_name : {}\t frame : {}\t set : {}\t read : {}\n'.format(video_path, video_name, frame_idx, is_set_, is_read_)
                    save_log(READ_LOG, os.path.join(save_root_dir, 'read_log.txt'))
                    
                    NON_READ_CNT+=1
                    continue


                # set dir / file name
                if truth == OOB_CLASS : # is out of body?
                    save_file_name = os.path.join(oob_save_dir, '{}_{:010d}.jpg'.format(video_name, frame_idx)) # OOB
                elif truth == IB_CLASS : # is inbody ?
                    save_file_name = os.path.join(ib_save_dir, '{}_{:010d}.jpg'.format(video_name, frame_idx)) # IB
                
                is_write_ = cv2.imwrite(save_file_name, img) # save

                # write sanity check
                if not (is_write_) :
                    print('video_path : ', video_path)
                    print('video_name : ', video_name)
                    print('frame : ', frame_idx)
                    print('file_name : ', save_file_name)
                    print('write :', is_write_)
                    
                    WRITE_LOG='video_path : {}\t video_name : {}\t frame : {}\t file_name : {}\t write : {}\n'.format(video_path, video_name, frame_idx, save_file_name, is_write_)
                    save_log(WRITE_LOG, os.path.join(save_root_dir, 'write_log.txt'))

                    NON_WRITE_CNT+=1
                    continue
                
                # processed count
                if truth == OOB_CLASS : # is out of body?
                    temp_OOB_cnt += 1
                elif truth == IB_CLASS : # is inbody ?
                    temp_IB_cnt += 1
    

            Processed_OOB_cnt+=temp_OOB_cnt
            Processed_IB_cnt+=temp_IB_cnt
            video.release()

            print('Video processing done | OOB : {:08d}, IB : {:08d}'.format(temp_OOB_cnt, temp_IB_cnt))
            log_txt='video_name : {}\t\tOOB:{}\t\tIB:{}\n'.format(video_name, temp_OOB_cnt, temp_IB_cnt)
            save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log
            
            
    print('Total processing done | OOB : {:08d}, IB : {:08d}'.format(Processed_OOB_cnt, Processed_IB_cnt))
    log_txt='DONE.\t\t\t\tPROCESSED OOB:{}\t\tPROCESSED IB:{}\n\n'.format(Processed_OOB_cnt, Processed_IB_cnt)
    log_txt+='NON READ : {} \t\t\t\t NON WRITE : {} \n\n'.format(NON_READ_CNT, NON_WRITE_CNT)
    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log

    # finish time stamp
    finishTime = time.time()
    f_tm = time.localtime(finishTime)

    log_txt='FINISHED AT : \t' + time.strftime('%Y-%m-%d %I:%M:%S %p \n', f_tm)
    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log


if __name__ == '__main__' :
    # video_root_dir = '/data/CAM_IO/robot/video'
    # anno_root_dir = '/data/CAM_IO/robot/OOB'
    # save_root_dir = '/data/CAM_IO/robot/OOB_images_temp'


    video_root_dir = '/data/LAPA/Video'
    anno_root_dir = '/data/OOB'
    save_root_dir = '/data/LAPA/Img_Anno'

    capture_step = 30 # frame
    # mode = ['ROBOT', 'LAPA']
    gen_image_dataset_for_oob(video_root_dir, anno_root_dir, save_root_dir, capture_step, mode='LAPA')