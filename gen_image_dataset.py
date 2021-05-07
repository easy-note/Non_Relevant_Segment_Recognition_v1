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
def gettering_information_for_robot (video_root_path, anno_root_path, video_set, fps=30, video_ext='.mp4') : # paring video from annotation info
    print('\n\n\n\t\t\t ### STARTING DEF [gettering_information_for_robot] ### \n\n')
    
    fps = 30

    info_dict = {
        'video': [],
        'anno': [],
    }

    all_video_path = glob.glob(video_root_path +'/*{}'.format(video_ext)) # all video file list
    all_anno_path = glob.glob(anno_root_path + '/*.csv') # all annotation file list    
    
    # dpath = os.path.join(video_root_path) # video_root path
    print('NUMBER OF TOTAL VIDEO FILE : ', len(all_video_path))
    print('NUMBER OF TOTAL ANNOTATION FILE : ', len(all_anno_path))
    print('')

    for video_no in video_set : # get target video
        video_path_list = sorted([vfile for vfile in all_video_path if os.path.basename(vfile).startswith(video_no)])
        anno_path_list = sorted([anno_file for anno_file in all_anno_path if os.path.basename(anno_file).startswith(video_no)])
        
        print('\t ==== GETTERING INFO ====')
        print('\t VIDEO NO | ', video_no) 
        print('\t video_path', video_path_list) # target videos path
        print('\t anno_path', anno_path_list) # target annotaion path
        print('\t ==== ==== ==== ====\n')

        # check not paring num
        assert len(video_path_list) == len(anno_path_list), 'CANNOT PARING DATA'

        # it will be append to info_dict
        target_video_list = []
        target_anno_list = []
        
        for target_video_dir, target_anno_dir in (zip(video_path_list, anno_path_list)) :

            
            ## check of each pair
            print('PARING SANITY CHECK ====> ', end='')

            temp_token = os.path.basename(target_anno_dir).split('_')[:-1]
            temp_token.pop(1) # pop 'CAMIO'

            if os.path.basename(target_video_dir) == '_'.join(temp_token) + video_ext :
                print('\t\t done')
                print(target_video_dir)
                print(target_anno_dir)
            else :
                print('fail')
                print(target_video_dir)
                print(target_anno_dir)
                exit(1) 

            ## check end ##

            # continue to paring
            anno_df = pd.read_csv(target_anno_dir)
            anno_df = anno_df.dropna(axis=0) # 결측행 제거

            print(anno_df)
            
            # it will be append to temp_anno_list
            target_idx_list = []
            

            # time -> frame idx
            for i in range(len(anno_df)) :
    
                t_start = anno_df.loc[i]['start']
                t_end = anno_df.loc[i]['end']
                
                target_idx_list.append([time_to_idx(t_start, fps), time_to_idx(t_end, fps)]) # temp_idx_list = [[start, end], [start, end]..]

            # save gettering info
            target_video_list.append(target_video_dir) # [video1_1, video_1_2, ...]
            target_anno_list.append(target_idx_list) # [temp_idx_list_1_1, temp_idx_list_1_2, ... ]

        # info_dict['video'], info_dict['anno'] length is same as valset
        info_dict['video'].append(target_video_list) # [[video1_1, video1_2], [video2_1, video_2_2], ...]
        info_dict['anno'].append(target_anno_list) # [[temp_idx_list_1_1, temp_idx_list_1_2], [temp_idx_list_2_1, temp_idx_list_2_2,], ...]
        
        print('\n\n')
        
    return info_dict


def gen_image_dataset_for_robot(video_root_dir, anno_root_dir, save_root_dir, capture_step):
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

    video_set = ['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R010', 'R013', 'R014', 'R015', 'R017', 'R018', 
                'R019', 'R022', 'R048', 'R056', 'R074', 'R076', 'R084', 'R094', 'R100', 'R116', 'R117', 'R201', 'R202', 'R203', 
                'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R301', 'R302', 'R303', 'R304', 'R305', 'R313']
    
    # paring video from annotation info
    info_dict = gettering_information_for_robot(video_root_dir, anno_root_dir, video_set, fps=30, video_ext='.mp4')

    total_videoset_cnt = len(info_dict['video']) # total number of video set
    print('{} VIDEO SET WILL BE CAPUERTRED ... '.format(total_videoset_cnt))

    Processed_OOB_cnt = 0
    Processed_IB_cnt = 0
    NON_READ_CNT = 0
    NON_WRITE_CNT = 0

    # loop from total_videoset_cnt
    for i, (video_path_list, anno_info_list) in enumerate(zip(info_dict['video'], info_dict['anno']), 1): 
        videoset_name = os.path.basename(video_path_list[0]).split('_')[0] # parsing videoset name

        print('COUNT OF VIDEO SET | {} / {} \t\t ======>  VIDEO SET | {}'.format(i, total_videoset_cnt, videoset_name))
        print('NUMBER OF VIDEO : {} | NUMBER OF ANNOTATION INFO : {}'.format(len(video_path_list), len(anno_info_list)))
        print('IMAGE SAVED AT \t\t\t ======>  {}'.format(save_root_dir))
        print('\n')

        for video_path, anno_info in zip(video_path_list, anno_info_list) :
            
            video_name = os.path.basename(video_path).split('.')[0] # only video name
            print(video_name)

            # open video cap
            video = cv2.VideoCapture(video_path)
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            print('\tTarget video : {} | Total Frame : {}'.format(video_name, video_len))
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
    video_root_dir = '/data/CAM_IO/robot/video'
    anno_root_dir = '/data/CAM_IO/robot/OOB'
    save_root_dir = '/data/CAM_IO/robot/OOB_images_temp'
    capture_step = 30 # 1sec
    gen_image_dataset_for_robot(video_root_dir, anno_root_dir, save_root_dir, capture_step)