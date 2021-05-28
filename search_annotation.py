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

            # open video cap, only check for frame count
            video = cv2.VideoCapture(target_video_dir)
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = video.get(cv2.CAP_PROP_FPS)
            video.release()

            # only target_video_dir 
            if target_anno_dir != '' :

                if mode=='ROBOT' : # csv
                    anno_df = pd.read_csv(target_anno_dir)
                    anno_df = anno_df.dropna(axis=0) # 결측행 제거

                    # time -> frame idx
                    for i in range(len(anno_df)) :
                        t_start = anno_df.iloc[i]['start']
                        t_end = anno_df.iloc[i]['end']
                        
                        target_idx_list.append([time_to_idx(t_start, fps), time_to_idx(t_end, fps)]) # temp_idx_list = [[start, end], [start, end]..]
                    
                    print('-----'*3)
                    print(target_video_dir)
                    print(target_anno_dir)
                    print(anno_df)

                
                elif mode=='LAPA' : # json
                    with open(target_anno_dir) as json_file:
                        json_data = json.load(json_file)

                    # annotation frame
                    for anno_data in json_data['annotations'] :
                        f_start = anno_data['start']
                        f_end = anno_data['end']
                        
                        # truncate
                        target_idx_list.append([f_start, f_end]) # temp_idx_list = [[start, end], [start, end]..]

                    print('-----'*3)
                    print(target_video_dir)
                    print(target_anno_dir)
                    print('ANNO \t FPS {} | TOTAL {}'.format(json_data['frameRate'], json_data['totalFrame']))
                    print('VIDEO \t FPS {} | TOTAL {}'.format(video_fps, video_len))
                    log_txt='VIDEO | {}\t\tANNO | {}\n'.format(os.path.splitext(os.path.basename(target_video_dir))[0], os.path.splitext(os.path.basename(target_anno_dir))[0])
                    log_txt+='ANNO \t FPS {} | TOTAL {}\n'.format(json_data['frameRate'], json_data['totalFrame'])
                    log_txt+='VIDEO \t FPS {} | TOTAL {}\n'.format(video_fps, video_len)
                    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log

                    if(json_data['totalFrame'] != video_len) :
                        print('NOT PAIR anno : {} | video : {}'.format(json_data['totalFrame'], video_len))
                        log_txt='NOT PAIR [TotalFrame]\t video : {} | anno : {} | video : {}\n'.format(os.path.splitext(os.path.basename(target_video_dir))[0], json_data['totalFrame'], video_len)
                        save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log

                    print(json_data['annotations'])
                    
                    save_log('\n----------------------\n', os.path.join(save_root_dir, 'log.txt')) # save log

                    
                    
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

def check_anno_sequence(anno_info:list):

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

def check_anno_over_frame(anno_info:list, video_len):
    has_not_over_frame = False
    
    last_start, last_end = anno_info[-1]
    
    if last_end < video_len : 
        has_not_over_frame = True
    
    print('ANNO LAST FRAME END : {} | VIDEO_LEN : {}'.format(last_end, video_len))

    return has_not_over_frame

def check_anno_float(anno_info:list):
    is_not_float = False 
    
    for start, end in anno_info : 
        if (int(start) == start) and (int(end) == end) :
            is_not_float = True
    
    return is_not_float


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
            os.makedirs(os.path.join(oob_save_dir, 'start'))
            os.makedirs(os.path.join(oob_save_dir, 'end'))
    except OSError :
        print('ERROR : Creating Directory, ' + oob_save_dir)

    try :
        if not os.path.exists(ib_save_dir) :
            os.makedirs(ib_save_dir)
            os.makedirs(os.path.join(ib_save_dir, 'start'))
            os.makedirs(os.path.join(ib_save_dir, 'end'))
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
        fps=30
        info_dict = gettering_information_for_oob(video_root_dir, anno_root_dir, robot_video_set, mode='ROBOT')
    elif mode=='LAPA' :
        fps=60
        info_dict = gettering_information_for_oob(video_root_dir, anno_root_dir, lapa_video_set, mode='LAPA')
    else :
        assert False, 'ONLY SUPPORT MODE [ROBOT, LAPA] | Input mode : {}'.format(mode) 
    
    total_videoset_cnt = len(info_dict['video']) # total number of video set
    print('{} VIDEO SET WILL BE CAPUERTRED ... '.format(total_videoset_cnt))

    Processed_OOB_cnt = 0
    Processed_IB_cnt = 0
    NON_READ_CNT = 0
    NON_WRITE_CNT = 0

    ### temp###
    total_margin_cnt = 0
    total_abnormal_margin_cnt = 0 

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
            if anno_info : # not empty list
                for start, end in anno_info :
                    print([idx_to_time(start, fps), idx_to_time(end, fps)])
            else : # empty
                print(anno_info)
                print('=====> NO EVENT')

            print('')


            ####  make truth list ####
            IB_CLASS, OOB_CLASS = [0, 1]
            truth_list = np.zeros(video_len, dtype='uint8') if IB_CLASS == 0 else np.ones(video_len, dtype='uint8')

            if anno_info : # only has event
                for start, end in anno_info :
                    truth_list[int(math.floor(start)):int(math.floor(end+1))] = OOB_CLASS # OOB Section # it's ok to out of range (numpy)

            truth_oob_count = truth_list.count(OOB_CLASS)

            print('IB_CLASS = {} | OOB_CLASS = {}'.format(IB_CLASS, OOB_CLASS))
            print('TRUTH IB FRAME COUNT : ', video_len - truth_oob_count)
            print('TRUTH OOB FRAME COUNT : ', truth_oob_count)
            ### ### ###

            temp_OOB_cnt = 0
            temp_IB_cnt = 0
            print('\n\n \t *** START *** \n\n')
            
            if anno_info : # only has event
                print(anno_info)
                # checking anno_seq
                if not(check_anno_sequence(anno_info)): 
                    save_log('\n{}-{}\n'.format(video_name, 'Annotation Sequence ERROR'), os.path.join(save_root_dir, 'log.txt')) # save log
                    print('\n{}-{}\n'.format(video_name, 'Annotation Sequence ERROR'))
                
                # checking anno has overframe
                if not(check_anno_over_frame(anno_info, video_len)) : 
                    save_log('\n{}-{}\n'.format(video_name, 'Annotation Last Frame ERROR'), os.path.join(save_root_dir, 'log.txt')) # save log
                    print('\n{}-{}\n'.format(video_name, 'Annotation Last Frame ERROR'))
                
                # checking anno float
                if not(check_anno_float(anno_info)) : 
                    save_log('\n{}-{}\n'.format(video_name, 'Annotation Float'), os.path.join(save_root_dir, 'log.txt')) # save log
                    print('\n{}-{}\n'.format(video_name, 'Annotation Float Frame ERROR'))

                continue

                # checking annotation margin
                for start_idx, end_idx in anno_info : 
                    # target_idx_list.append([int(math.floor(t_start)), int(math.floor(t_end))]) # temp_idx_list = [[start, end], [start, end]..]
                    
                    end_idx = end_idx-1 # last frame exception
                    # refined 
                    refined_start_idx = int(math.floor(start_idx))
                    refined_end_idx = int(math.floor(end_idx))
                    
                    print('video_name : {}\t\t video_len : {}\t\t start:{}\t\t end:{}\t\t'.format(video_name, video_len, start_idx, end_idx))

                    if (start_idx != refined_start_idx) and (end_idx != refined_end_idx) :
                        total_abnormal_margin_cnt += 1

                    start_class = truth_list[refined_start_idx]
                    end_class = truth_list[refined_end_idx]

                    log_txt='video_name : {}\t\t video_len : {}\t\t start:{}\t refined_start:{}\t s_class:{}\t\t end:{}\t refined_end:{}\t e_class:{}\t\n'.format(video_name, video_len, start_idx, refined_start_idx, start_class, end_idx, refined_end_idx, end_class)
                    print(log_txt)
                    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log

                    # start capture
                    is_set_= None
                    is_read_= None
                    img = None
                    print(video.set(cv2.CAP_PROP_FPS, float(30.0)))
                    is_set_ = video.set(cv2.CAP_PROP_POS_FRAMES, video_len-1) # frame setting
                    is_read_, img = video.read() # read frame

                
                    # set dir / file name // start_class
                    if start_class == OOB_CLASS : # is out of body?
                        save_file_name = os.path.join(oob_save_dir, 'start', '{}_{:010d}.jpg'.format(video_name, refined_start_idx)) # OOB
                    elif start_class == IB_CLASS : # is inbody ?
                        save_file_name = os.path.join(ib_save_dir,'start', '{}_{:010d}.jpg'.format(video_name, refined_start_idx)) # IB

                    if not (is_set_ and is_read_) :
                        READ_LOG='video_path : {}\t video_name : {}\t total_frame : {}\t frame : {}\t set : {}\t read : {} \t\t[start]\n'.format(video_path, video_name, video_len, video_len, is_set_, is_read_)
                        save_log(READ_LOG, os.path.join(save_root_dir, 'read_log.txt'))
                    else :
                        is_write_ = cv2.imwrite(save_file_name, img) # save

                    # end capture
                    is_set_= None
                    is_read_= None
                    img = None
                    is_set_ = video.set(cv2.CAP_PROP_POS_FRAMES, video_len) # frame setting
                    is_read_, img = video.read() # read frame



                    if start_class == OOB_CLASS : # is out of body?
                        save_file_name = os.path.join(oob_save_dir, 'end', '{}_{:010d}.jpg'.format(video_name, refined_end_idx)) # OOB
                    elif end_class == IB_CLASS : # is inbody ?
                        save_file_name = os.path.join(ib_save_dir,'end', '{}_{:010d}.jpg'.format(video_name, refined_end_idx)) # IB
                    
                    if not (is_set_ and is_read_) :
                        READ_LOG='video_path : {}\t video_name : {}\t total_frame : {}\t frame : {}\t set : {}\t read : {} \t\t[end]\n'.format(video_path, video_name, video_len, video_len, is_set_, is_read_)
                        save_log(READ_LOG, os.path.join(save_root_dir, 'read_log.txt'))
                    else :
                        is_write_ = cv2.imwrite(save_file_name, img) # save

                    total_margin_cnt+=1
    

            Processed_OOB_cnt+=temp_OOB_cnt
            Processed_IB_cnt+=temp_IB_cnt
            video.release()

            # print('Video processing done | OOB : {:08d}, IB : {:08d}'.format(temp_OOB_cnt, temp_IB_cnt))
            # log_txt='video_name : {}\t\tOOB:{}\t\tIB:{}\n'.format(video_name, temp_OOB_cnt, temp_IB_cnt)
            # save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log
    exit(0)
            
    print('Total processing done | OOB : {:08d}, IB : {:08d}'.format(Processed_OOB_cnt, Processed_IB_cnt))
    log_txt='DONE.\t\t\t\tPROCESSED OOB:{}\t\tPROCESSED IB:{}\n\n'.format(Processed_OOB_cnt, Processed_IB_cnt)
    log_txt+='NON READ : {} \t\t\t\t NON WRITE : {} \n\n'.format(NON_READ_CNT, NON_WRITE_CNT)
    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log

    ##### dummy
    log_txt='TOTAL MARGIN CNT : {}\t ABNOTMAL MARGIN CNT : {}\t\n\n'.format(total_margin_cnt, total_abnormal_margin_cnt)
    save_log(log_txt, os.path.join(save_root_dir, 'log.txt')) # save log
    ### 


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

    capture_step = 60 # frame
    # mode = ['ROBOT', 'LAPA']
    gen_image_dataset_for_oob(video_root_dir, anno_root_dir, save_root_dir, capture_step, mode='LAPA')