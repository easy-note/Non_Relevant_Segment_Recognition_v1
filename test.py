"""
For model test using pre-created test datset in tensor form.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib
import argparse
import time
import json
import datetime

from pandas import DataFrame as df
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import torch
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from train_model import CAMIO
from test_info_dict import gettering_information_for_oob
from test_info_dict import sanity_check_info_dict

matplotlib.use('Agg')


parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, help='trained model_path')
parser.add_argument('--data_dir', type=str,
                    default='/data/ROBOT/Video', help='video_path :) ')
parser.add_argument('--anno_dir', type=str,
                    default='/data/OOB', help='annotation_path :) ')
parser.add_argument('--results_save_dir', type=str, help='inference results save path')

parser.add_argument('--mode', type=str,
                    default='ROBOT', choices=['ROBOT', 'LAPA'], help='inference results save path')

## trained model (you should put same model as trained model)
parser.add_argument('--model', type=str,
                    choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d', 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'squeezenet1_0'], help='trained backborn model')
# inference frame step
parser.add_argument('--inference_step', type=int, default=5, help='inference frame step')

# inference video
parser.add_argument('--test_videos', type=str, nargs='+',
                    choices=['R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_10', 'R_13', 'R_14', 'R_15', 'R_17', 'R_18', 
                            'R_19', 'R_22', 'R_48', 'R_56', 'R_74', 'R_76', 'R_84', 'R_94', 'R_100', 'R_116', 'R_117', 'R_201', 'R_202', 'R_203', 
                            'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_301', 'R_302', 'R_303', 'R_304', 'R_305', 'R_313'],
                    help='inference video')

# infernece assets root path
parser.add_argument('--inference_assets_dir', type=str, help='inference assets root path')

args, _ = parser.parse_known_args()


# check results for binary metric
def calc_confusion_matrix(gts, preds):
    IB_CLASS, OOB_CLASS = [0, 1]
    
    classification_report_result = classification_report(gts, preds, labels=[IB_CLASS, OOB_CLASS], target_names=['IB', 'OOB'], zero_division=0)
    
    prec = precision_score(gts, preds, average='binary',pos_label=1, zero_division=0) # pos = [1]
    recall = recall_score(gts, preds, average='binary',pos_label=1, zero_division=0) # pos = [1]

    metric = pd.crosstab(pd.Series(gts), pd.Series(preds), rownames=['True'], colnames=['Predicted'], margins=True)
    
    saved_text = '{} \nprecision \t : \t {} \nrecall \t\t : \t {} \n\n{}'.format(classification_report_result, prec, recall, metric)

    return saved_text

def idx_to_time(idx, fps) :
    time_s = idx // fps
    frame = idx % fps

    converted_time = str(datetime.timedelta(seconds=time_s))
    converted_time = converted_time + ':' + str(frame)

    return converted_time

# input|DataFrame = 'frame' 'time' truth' 'predict'
# calc FN, FP, TP, TN
# out|{} = FN, FP, TP, TN frame
def return_metric_frame(result_df) :
    IB_CLASS, OOB_CLASS = 0,1

    print(result_df)
    
    # FN    
    FN_df = result_df[(result_df['truth']==OOB_CLASS) & (result_df['predict']==IB_CLASS)]
    
    # FP
    FP_df = result_df[(result_df['truth']==IB_CLASS) & (result_df['predict']==OOB_CLASS)]

    # TN
    TN_df = result_df[(result_df['truth']==IB_CLASS) & (result_df['predict']==IB_CLASS)]
    
    # TP
    TP_df = result_df[(result_df['truth']==OOB_CLASS) & (result_df['predict']==OOB_CLASS)]

    return {
        'FN_df' : FN_df,
        'FP_df' : FP_df,
        'TN_df' : TN_df,
        'TP_df' : TP_df,
    }

# save video frame from frame_list | it will be saved in {save_path}/{video_name}-{frame_idx}.jpg
def save_video_frame(video_path, frame_list, save_path, video_name) :
    video = cv2.VideoCapture(video_path)
    
    print('TARGET VIDEO_PATH : ', video_path)
    print('TARGET FRAME : ', frame_list)

    for frame_idx in tqdm(frame_list, desc='Saving Frame From {} ... '.format(video_path)) :
        video.set(1, frame_idx) # frame setting
        _, img = video.read() # read frame

        cv2.imwrite(os.path.join(save_path, '{}-{:010d}.jpg'.format(video_name, frame_idx)), img)
    
    video.release()
    print('======> DONE.')

# calc OOB_false Metric
def calc_OOB_metric(FN_cnt, FP_cnt, TN_cnt, TP_cnt, TOTAL_cnt) :
    OOB_metric = -1

    try : # zero devision except
        OOB_metric = (TP_cnt - FP_cnt) / (FP_cnt + TP_cnt + FN_cnt) # positie = OOB
    except :
        OOB_metric = -1

    return OOB_metric

# save log 
def save_log(log_txt, save_dir) :
    print('=========> SAVING LOG ... | {}'.format(save_dir))
    with open(save_dir, 'a') as f :
        f.write(log_txt)


def test_start() :
    """
    - Logging.
    - Start test code.
    """
    ### ### create results folder for save args and log.txt ### ###
    try :
        if not os.path.exists(args.results_save_dir) :
            os.makedirs(args.results_save_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + args.results_save_dir)

    # save args log
    with open(os.path.join(args.results_save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log_txt='\n\n=============== \t\t COMMAND ARGUMENT \t\t ============= \n\n'
    log_txt+=json.dumps(args.__dict__, indent=2)

    # start time stamp
    startTime = time.time()
    s_tm = time.localtime(startTime)
    
    log_txt+='\n\n=============== \t\t INFERNECE TIME \t\t ============= \n\n'
    log_txt+='STARTED AT : \t' + time.strftime('%Y-%m-%d %I:%M:%S %p \n', s_tm)
    
    save_log(log_txt, os.path.join(args.results_save_dir, 'log.txt')) # save log

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    
    '''
    # load args log
    parser = ArgumentParser()
    args = parser.parse_args()
    with open('commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    '''
    
    # inference setting for each mode
    test_hparams = {
        'optimizer_lr' : 0, # dummy (this option use only for training)
        'backborn_model' : args.model # (for train, test)
    }

    print('')
    print('')
    print(test_hparams)

    # model 불러오기.
    model = CAMIO.load_from_checkpoint(args.model_path, config=test_hparams)

    model.cuda()

    print('\n\t=== model_loded for {} ===\n'.format(args.mode))

    if args.mode == 'LAPA' : 
        # starting inference
        test_for_lapa(args.data_dir, args.anno_dir, args.inference_assets_dir, args.results_save_dir, model, args.test_videos, args.inference_step)

    else: # robot
        # starting inference
        test_for_robot(args.data_dir, args.anno_dir, args.inference_assets_dir, args.results_save_dir, model, args.test_videos, args.inference_step)

    # finish time stamp
    finishTime = time.time()
    f_tm = time.localtime(finishTime)

    log_txt = 'FINISHED AT : \t' + time.strftime('%Y-%m-%d %I:%M:%S %p \n', f_tm)
    save_log(log_txt, os.path.join(args.results_save_dir, 'log.txt')) # save log

def test_for_robot(data_dir, anno_dir, infernece_assets_dir, results_save_dir, model, patient_list, inference_step):
    """
    - Create info_dict (gettering_information_for_oob, sanity_check_info_dict) 
    - Execute test.py 

    Args:
        data_dir: Video root path.
        anno_dir: Annotation root path.
        infernece_assets_dir: Inference assets root path.
        results_save_dir: Path for save directory.
        model: Backborn model.
        patient_list: patients to test.
        inference_step: Frame unit for test.
    """
    
    ### base setting ###
    # val_videos = ['R_17', 'R_22', 'R_116', 'R_208', 'R_303']
    valset = patient_list
    fps = 30

    # gettering information step
    info_dict = gettering_information_for_oob(data_dir, anno_dir, infernece_assets_dir, valset, mode='ROBOT')

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print('\tSUCESS GETTERING VIDEO SET: ', len(info_dict['video']))
    print('\tSUCESS GETTERING ANNOTATION SET: ', len(info_dict['anno']))
    print('\tSUCESS GETTERING INFERENCE SET: ', len(info_dict['inference_assets']))
    print('\t=== === === ===\n\n')

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print(info_dict['video'])
    print(info_dict['anno'])
    print(info_dict['inference_assets'])
    print('\t=== === === ===\n\n')

    #### sanity check and modify info_dict ###
    info_dict = sanity_check_info_dict(info_dict)

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print('\tSUCESS GETTERING VIDEO SET: ', len(info_dict['video']))
    print('\tSUCESS GETTERING ANNOTATION SET: ', len(info_dict['anno']))
    print('\tSUCESS GETTERING INFERENCE SET: ', len(info_dict['inference_assets']))
    print('\t=== === === ===\n\n')

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print(info_dict['video'])
    print(info_dict['anno'])
    print(info_dict['inference_assets'])
    print('\t=== === === ===\n\n')

    # inference step
    test(info_dict, model, results_save_dir, inference_step, fps=fps)

def test_for_lapa(data_dir, anno_dir, infernece_assets_dir, results_save_dir, model, patient_list, inference_step):
    """
    - Create info_dict (gettering_information_for_oob, sanity_check_info_dict) 
    - Execute test.py 

    Args:
        data_dir: Video root path.
        anno_dir: Annotation root path.
        infernece_assets_dir: Inference assets root path.
        results_save_dir: Path for save directory.
        model: Backborn model.
        patient_list: patients to test.
        inference_step: Frame unit for test.
    """
    
    ### base setting ###
    # val_videos = ['R_17', 'R_22', 'R_116', 'R_208', 'R_303']
    valset = patient_list
    fps = 60

    # gettering information step
    info_dict = gettering_information_for_oob(data_dir, anno_dir, infernece_assets_dir, valset, mode='LAPA')

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print('\tSUCESS GETTERING VIDEO SET: ', len(info_dict['video']))
    print('\tSUCESS GETTERING ANNOTATION SET: ', len(info_dict['anno']))
    print('\tSUCESS GETTERING INFERENCE SET: ', len(info_dict['inference_assets']))
    print('\t=== === === ===\n\n')

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print(info_dict['video'])
    print(info_dict['anno'])
    print(info_dict['inference_assets'])
    print('\t=== === === ===\n\n')

    #### sanity check and modify info_dict ###
    info_dict = sanity_check_info_dict(info_dict)

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print('\tSUCESS GETTERING VIDEO SET: ', len(info_dict['video']))
    print('\tSUCESS GETTERING ANNOTATION SET: ', len(info_dict['anno']))
    print('\tSUCESS GETTERING INFERENCE SET: ', len(info_dict['inference_assets']))
    print('\t=== === === ===\n\n')

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print(info_dict['video'])
    print(info_dict['anno'])
    print(info_dict['inference_assets'])
    print('\t=== === === ===\n\n')

    # inference step
    test(info_dict, model, results_save_dir, inference_step, fps=fps)

def test(info_dict, model, results_save_dir, inference_step, fps=30) : # project_name is only for use for title in total_metric_df.csv
    """
    - Model test (inference).

    Args:
        info_dict: The final form of 'info_dict'.
            info_dict = {
            'video': [video1_path, video2_path, ... ],
            'anno': [ 1-[[start, end],[start, end]], 2-[[start,end],[start,end]], 3-... ],
            'inference_assets' : [test_dataset1_path, test_dataset2_path, ...]
            }
        results_save_dir: Path for save directory.
        inference_step: Frame unit for test.
        fps: Video fps.
    """

    print('\n\n\n\t\t\t ### STARTING DEF [test] ### \n\n')

    # create results folder
    try :
        if not os.path.exists(results_save_dir) :
            os.makedirs(results_save_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + results_save_dir)

    total_videoset_cnt = len(info_dict['video']) # total number of video set

    # init total metric df
    total_metric_df = pd.DataFrame(index=range(0, 0), columns=['Video_set', 'Video_name', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'Confidence_Ratio']) # row cnt is same as checking vidoes length
    patient_total_metric_df = pd.DataFrame(index=range(0, 0), columns=['Patient', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'Confidence_Ratio']) # row cnt is same as total_videoset_cnt

    # loop from total_videoset_cnt
    for i, (video_path_list, anno_info_list, infernece_assets_path_list) in enumerate(zip(info_dict['video'], info_dict['anno'], info_dict['inference_assets']), 1):
        
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice = os.path.splitext(os.path.basename(video_path_list[0]))[0].split('_') # parsing videoset name
        videoset_name = '{}_{}'.format(op_method, patient_idx)

        # init for patient results_dict
        patient_video_list = []
        patient_frame_idx_list = []
        patient_time_list = []
        patient_gt_list = []
        patient_predict_list = []
        patient_truth_oob_count = 0
        patient_truth_ib_count = 0

        # create base folder for save results each video set
        each_videoset_result_dir = os.path.join(results_save_dir, videoset_name) # '~~~/results/R022' , '~~~/results/R011' ..
        try :
            if not os.path.exists(os.path.join(each_videoset_result_dir)) :
                os.makedirs(each_videoset_result_dir)
        except OSError :
            print('ERROR : Creating Directory, ' + each_videoset_result_dir)


        print('COUNT OF VIDEO SET | {} / {} \t\t ======>  VIDEO SET | {}'.format(i, total_videoset_cnt, videoset_name))
        print('NUMBER OF VIDEO : {} | NUMBER OF ANNOTATION INFO : {}'.format(len(video_path_list), len(anno_info_list)))
        print('NUMBER OF INFERNECE ASSETS {} \n==> ASSETS LIST : {}'.format(len(infernece_assets_path_list), infernece_assets_path_list))
        print('RESULTS SAVED AT \t\t\t ======>  {}'.format(each_videoset_result_dir))
        print('\n')
        
        # extract info for each video_path
        for video_path, anno_info, each_video_infernece_assets_path_list in zip(video_path_list, anno_info_list, infernece_assets_path_list) :
            
            video_name = os.path.splitext(os.path.basename(video_path))[0] # only video name

            # inference results saved folder for each video
            each_video_result_dir = os.path.join(each_videoset_result_dir, video_name) # '~~~/results/R022/R022_ch1_video_01' , '~~~/results/R022/R022_ch1_video_04' ..
            try :
                if not os.path.exists(os.path.join(each_video_result_dir)) :
                    os.makedirs(each_video_result_dir)
            except OSError :
                print('ERROR : Creating Directory, ' + each_video_result_dir)

            # FP Frame saved folder for each video
            fp_frame_saved_dir = os.path.join(each_video_result_dir, 'fp_frame') # '~~~/results/R022/R022_ch1_video_01/fp_frame'
            try :
                if not os.path.exists(os.path.join(fp_frame_saved_dir)) :
                    os.makedirs(fp_frame_saved_dir)
            except OSError :
                print('ERROR : Creating Directory, ' + fp_frame_saved_dir)
            
            # FN Frame saved folder for each video
            fn_frame_saved_dir = os.path.join(each_video_result_dir, 'fn_frame') # '~~~/results/R022/R022_ch1_video_01/fn_frame'
            try :
                if not os.path.exists(os.path.join(fn_frame_saved_dir)) :
                    os.makedirs(fn_frame_saved_dir)
            except OSError :
                print('ERROR : Creating Directory, ' + fn_frame_saved_dir)

            # open video cap, only check for frame count
            video = cv2.VideoCapture(video_path)
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = video.get(cv2.CAP_PROP_FPS)
            video.release()

            print('\tTarget video : {} | Total Frame : {} | Video FPS : {} '.format(video_name, video_len, video_fps))
            print('\tAnnotation Info : {}'.format(anno_info))

            ### check idx -> time
            if anno_info : # event 
                for start, end in anno_info :
                    print([idx_to_time(start, fps), idx_to_time(end, fps)])
            else : # no evnet
                print(anno_info)
                print('=====> NO EVENT')
                pass
                
            ###
            print('')

            ####  make truth list ####
            IB_CLASS, OOB_CLASS = [0, 1]
            truth_list = np.zeros(video_len, dtype='uint8') if IB_CLASS == 0 else np.ones(video_len, dtype='uint8')

            if anno_info : # only has event
                for start, end in anno_info :
                    truth_list[start:end+1] = OOB_CLASS # OOB Section

            truth_list = list(truth_list) # change to list

            truth_oob_count = truth_list.count(OOB_CLASS)
            truth_ib_count = truth_list.count(IB_CLASS)

            print('IB_CLASS = {} | OOB_CLASS = {}'.format(IB_CLASS, OOB_CLASS))
            print('TRUTH IB FRAME COUNT : ', video_len - truth_oob_count)
            print('TRUTH OOB FRAME COUNT : ', truth_oob_count)
            ### ### ###

            # init_variable
            gt_list = [] # ground truth
            predict_list = [] # predict
            frame_idx_list = [] # frmae info
            time_list = [] # frame to time

            FP_frame_cnt = 0
            FN_frame_cnt = 0
            TP_frame_cnt = 0
            TN_frame_cnt = 0
            TOTAL_frame_cnt = 0
            # frame_check_cnt = 0 # loop cnt
            OOB_metric = -1 # false metric

            print('----')
            print('EACH INFERENCE ASEETS PATH LENGTH')
            inference_assets_cnt = len(each_video_infernece_assets_path_list)
            print(inference_assets_cnt)

            # split gt_list per inference assets
            # init_variable
            SLIDE_FRAME = 30000
            inference_assets_start_end_frame_idx= [[start_idx*SLIDE_FRAME, (start_idx+1)*SLIDE_FRAME-1] for start_idx in range(inference_assets_cnt)]
            inference_assets_start_end_frame_idx[-1][1] = video_len - 1 # last frame of last pickle
            print(inference_assets_start_end_frame_idx)

            inference_frame_idx_list = [] # inference target frame [[pickle_0], [pickle_1], [pickle_2],...]

            for start, end in inference_assets_start_end_frame_idx :
                inference_frame_idx_list.append([idx for idx in range(start, end+1) if idx % inference_step == 0])
                
            BATCH_SIZE = 512

            temp_cnt = 0 ### dummy for test

            # getting infernce aseets 
            for infernece_assests_path, inference_frame_idx, (start_idx, end_idx) in zip(each_video_infernece_assets_path_list, inference_frame_idx_list, inference_assets_start_end_frame_idx) :
                print('\n\n=============== \tTARGET INFRENCE \t ============= \n\n')
                print('======> VIDEO PATH')
                print(video_path)

                print('======> ANNO INFO')
                print(anno_info)

                print('======> TRUTH_LIST LENGTH')
                print(len(truth_list))

                print('======> NUMBER OF INFERENCE ASSETS')
                print(len(each_video_infernece_assets_path_list))

                print('======> INFERENCE ASSETS PATH')
                print(infernece_assests_path)

                print('======> INFERENCE FRAME IDX [START, END]')
                print([start_idx, end_idx])

                print('======> INFERENCE FRAME IDX ')
                print(inference_frame_idx)

                print('')
                
                print('\n\n=============== \t\t ============= \n\n')

                # load inference assets from pickle
                print('===> LOADING... | \t {} '.format(infernece_assests_path))
                if True :
                    with open(infernece_assests_path, 'rb') as file :    
                        infernce_asset = pickle.load(file)
                        print('\t ===> DONE')
                else : # for test
                    infernce_asset = torch.Tensor(inference_assets_start_end_frame_idx[temp_cnt][-1], 3, 224, 224) # zero, float32
                    temp_cnt += 1
                print('\n\n=============== \t\t ============= \n\n')
                
                
                # infernceing 
                with torch.no_grad() : #autograd 끔 - 메모리 사용량 줄이고, 연산 속도 높힘. 사실상 안 쓸 gradient 라서 inference 시에 굳이 계산할 필요 없음. 

                    model.eval() # 21.05.30 JH 추가 - layer 의 동작을 inference(eval) mode로 변경, 모델링 시 training과 inference 시에 다르게 동작하는 layer 들이 존재함. (e.g. Dropout layer, BatchNorm)

                    # batch slice from infernece_frame_idx
                    start_pos = 0
                    end_pos = len(inference_frame_idx)
                    
                    FRAME_INDICES = []

                    # inferencing per batch
                    for idx in tqdm(range(start_pos, end_pos + BATCH_SIZE, BATCH_SIZE),  desc='Inferencing... \t ==> {} | {}'.format(video_name, infernece_assests_path)):
                        FRAME_INDICES = inference_frame_idx[start_pos:start_pos + BATCH_SIZE] # video frame idx
                        if FRAME_INDICES != []:
                            TORCH_INDIES = [f_idx-start_idx for f_idx in FRAME_INDICES] # convert to torch idx

                            # make batch input tensor
                            # index_select 함수를 사용해 지정한 차원 기준으로 (0) 원하는 값들을 뽑아낼 수 있음.
                            BATCH_INPUT_TENSOR = torch.index_select(infernce_asset, 0, torch.tensor(TORCH_INDIES)) #, 21.05.30 JH 수정 - ERROR dtype=torch.int32 - dtype 을 바꿔도 되는지 확인 필요. (int64로 변경됨.)
                            
                            # upload on cuda
                            BATCH_INPUT_TENSOR = BATCH_INPUT_TENSOR.cuda()

                            # inferencing model
                            outputs = model(BATCH_INPUT_TENSOR)

                            predict = torch.argmax(outputs.cpu(), 1) # predict 
                            predict = predict.tolist() # tensot -> list
                            # print(predict)

                            # save results
                            frame_idx_list+=FRAME_INDICES
                            predict_list+= list(predict)
                            
                            gt_list+=(lambda in_list, indices_list : [in_list[i] for i in indices_list])(truth_list, FRAME_INDICES) # get in_list elements from indices index 
                            time_list+=[idx_to_time(frame_idx, fps) for frame_idx in FRAME_INDICES] # FRAME INDICES -> time


                        start_pos = start_pos + BATCH_SIZE
                

                del infernce_asset # free memory

            print('\n\n=============== \t\t ============= \n\n')
            print('\t ===> INFERNECE DONE')
            print('\n\n=============== \t\t ============= \n\n')
            
            print(frame_idx_list)
            print(predict_list)
            print(gt_list)
            print(time_list)

            # saving inferece result
            result_dict = {
                'frame' : frame_idx_list,
                'time' : time_list,
                'truth' : gt_list,
                'predict' : predict_list
            }

            # append inference results for patient
            patient_video_list.append(video_name)
            patient_frame_idx_list.append(frame_idx_list)
            patient_time_list.append(time_list)
            patient_gt_list.append(gt_list)
            patient_predict_list.append(predict_list)
            patient_truth_oob_count+=truth_oob_count
            patient_truth_ib_count+=truth_ib_count

            # save for df
            inference_results_df = pd.DataFrame(result_dict)
        
            print('Result Saved at \t ====> ', each_video_result_dir)
            inference_results_df.to_csv(os.path.join(each_video_result_dir, 'Inference-{}-{}.csv'.format(args.mode, video_name)), mode="w")
            
            # calc FN, FP, TP, TN frame and TOTAL
            metric_frame = return_metric_frame(inference_results_df)

            FN_frame_cnt = len(metric_frame['FN_df'])
            FP_frame_cnt = len(metric_frame['FP_df'])
            TN_frame_cnt = len(metric_frame['TN_df'])
            TP_frame_cnt = len(metric_frame['TP_df'])
            TOTAL_frame_cnt = len(inference_results_df)

            print('\n\n=============== \tFN\t ============= \n\n')
            print(metric_frame['FN_df'])

            print('\n\n=============== \tFP\t ============= \n\n')
            print(metric_frame['FP_df'])
            

            print('\n\n=============== \tTN\t ============= \n\n')
            print(metric_frame['TN_df'])
            

            print('\n\n=============== \tTP\t ============= \n\n')
            print(metric_frame['TP_df'])
            

            print('\n\n=============== \tINFO\t ============= \n\n')
            print(FN_frame_cnt)
            print(FP_frame_cnt)
            print(TN_frame_cnt)
            print(TP_frame_cnt)
            print(TOTAL_frame_cnt)
            print(truth_oob_count)
            print(video_len - truth_oob_count)
            print('\n\n=============== \t\t ============= \n\n')
            
            
            ### saving FP TN frame
            # FP frame
            print('\n\n=============== \tSAVE FP frame \t ============= \n\n')
            save_video_frame(video_path, list(metric_frame['FP_df']['frame']), fp_frame_saved_dir, video_name)

            # FN frame
            print('\n\n=============== \tSAVE FN frame \t ============= \n\n')
            save_video_frame(video_path, list(metric_frame['FN_df']['frame']), fn_frame_saved_dir, video_name)

            # OOB_Metric
            OOB_metric = calc_OOB_metric(FN_frame_cnt, FP_frame_cnt, TN_frame_cnt, TP_frame_cnt, TOTAL_frame_cnt)

            # saving FN FP TP TN Metric
            results_metric = {
                'Video_set' : [videoset_name],
                'Video_name' : [video_name],
                'FP' : [FP_frame_cnt],
                'TP' : [TP_frame_cnt],
                'FN' : [FN_frame_cnt],
                'TN' : [TN_frame_cnt],
                'TOTAL' : [TOTAL_frame_cnt],
                'GT_OOB' : [gt_list.count(OOB_CLASS)],
                'GT_IB' : [gt_list.count(IB_CLASS)],
                'PREDICT_OOB' : [predict_list.count(OOB_CLASS)],
                'PREDICT_IB' : [predict_list.count(IB_CLASS)],
                'GT_OOB_1FPS' : [truth_oob_count],
                'GT_IB_1FPS' : [video_len-truth_oob_count],
                'Confidence_Ratio' : [OOB_metric]
            }

            # each metric per video
            result_metric_df = pd.DataFrame(results_metric)
            
            # save
            print('Metric Saved at \t ====> ', each_video_result_dir)
            result_metric_df.to_csv(os.path.join(each_video_result_dir, 'Confidence_Ratio-{}-{}.csv'.format(args.mode, video_name)), mode="w")

            # append metric
            # columns=['Video_set', 'Video_name', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'Confidence_Ratio'])           
            # total_metric_df = total_metric_df.append(result_metric_df)
            total_metric_df = pd.concat([total_metric_df, result_metric_df], ignore_index=True) # shoul shink columns

            print('')
            print(total_metric_df)
            total_metric_df.to_csv(os.path.join(results_save_dir, 'Video_Total_metric-{}-{}.csv'.format(args.mode, os.path.basename(results_save_dir))), mode="w") # save on project direc

            # saving plot
            fig = plt.figure(figsize=(16,8))

            # plt.hold()
            plt.scatter(np.array(frame_idx_list), np.array(gt_list), color='blue', marker='o', s=20, label='Truth') # ground truth
            plt.scatter(np.array(frame_idx_list), np.array(predict_list), color='red', marker='o', s=5, label='Predict') # predict

            plt.title('Inference Results By per {} Frame | Video : {} | Results : {} '.format(inference_step, video_name, os.path.basename(results_save_dir)));
            plt.suptitle('OOB_CLASS [{}] | IB_CLASS [{}] | FP : {} | TP : {} | FN : {} | TN : {} | TOTAL : {} | Confidence_Ratio : {} '.format(OOB_CLASS, IB_CLASS, FP_frame_cnt, TP_frame_cnt, FN_frame_cnt, TN_frame_cnt, TOTAL_frame_cnt, OOB_metric));
            plt.ylabel('class'); plt.xlabel('frame');
            plt.legend(loc='center right');

            plt.savefig(os.path.join(each_video_result_dir, 'plot_{}.png'.format(video_name)));

            # save confusion matrix
            # def calc_confusion_matrix(inference_results_df['truth'], inference_results_df['predict'])
            saved_text = calc_confusion_matrix(gt_list, predict_list)
            saved_text += '\n\nFP\t\tTP\t\tFN\t\tTN\t\tTOTAL\n'
            saved_text += '{}\t\t{}\t\t{}\t\t{}\t\t{}\n\n'.format(FP_frame_cnt, TP_frame_cnt, FN_frame_cnt, TN_frame_cnt, TOTAL_frame_cnt)
            saved_text += 'Confidence_Ratio : {:.4f}'.format(OOB_metric)

            with open(os.path.join(each_video_result_dir, 'Metric-{}-{}.txt'.format(args.mode, video_name)), 'w') as f :
                f.write(saved_text)

            print('\n\n-------------- \t\t -------------- \n\n')

        print('\n\n=============== \t\t PATIENT PROCESSING \t\t ============= \n\n')
        
        #####  calc for patient #### 
        patient_FN_frame_cnt = 0
        patient_FP_frame_cnt = 0
        patient_TN_frame_cnt = 0
        patient_TP_frame_cnt = 0
        patient_TOTAL_frame_cnt = 0

        # video_name	frame	consensus_frame	time	consensus_time	truth	predict
        patient_inference_results_df = df(range(0,0), columns=['video_name', 'frame', 'consensus_frame', 'time', 'consensus_time', 'truth', 'predict'])

        # make results for patient
        for patient_video, patient_frame_idx, patient_time, patient_gt, patient_predict in zip(patient_video_list, patient_frame_idx_list, patient_time_list, patient_gt_list, patient_predict_list) :
            # saving inferece result per patient
            temp_patient_result_dict = {
                'video_name' : [patient_video]*len(patient_gt),
                'frame' : patient_frame_idx,
                'time' : patient_time,
                'truth' : patient_gt,
                'predict' : patient_predict
            }

            # re-index time and frame_idx
            patient_inference_results_df = pd.concat([patient_inference_results_df, df(temp_patient_result_dict)], ignore_index=True)

        # re-index consensus_frame
        patient_inference_results_df['consensus_frame'] = [frame * inference_step for frame in range(len(patient_inference_results_df))]
        # calc consensus time
        patient_inference_results_df['consensus_time'] = patient_inference_results_df.apply(lambda x : idx_to_time(x['consensus_frame'], fps), axis=1)

        # save
        print('Patient Result Saved at \t ====> ', each_videoset_result_dir)
        patient_inference_results_df.to_csv(os.path.join(each_videoset_result_dir, 'Inference-{}-{}.csv'.format(args.mode, videoset_name)), mode="w")

        # each metric per patient
        result_metric_df_per_patient = return_metric_frame(patient_inference_results_df)

        patient_FN_frame_cnt = len(result_metric_df_per_patient['FN_df'])
        patient_FP_frame_cnt = len(result_metric_df_per_patient['FP_df'])
        patient_TN_frame_cnt = len(result_metric_df_per_patient['TN_df'])
        patient_TP_frame_cnt = len(result_metric_df_per_patient['TP_df'])
        patient_TOTAL_frame_cnt = len(patient_inference_results_df)

        print('\n\n=============== \tFN\t ============= \n\n')
        print(result_metric_df_per_patient['FN_df'])

        print('\n\n=============== \tFP\t ============= \n\n')
        print(result_metric_df_per_patient['FP_df'])
        

        print('\n\n=============== \tTN\t ============= \n\n')
        print(result_metric_df_per_patient['TN_df'])
        

        print('\n\n=============== \tTP\t ============= \n\n')
        print(result_metric_df_per_patient['TP_df'])
        

        print('\n\n=============== \tINFO\t ============= \n\n')
        print(patient_FN_frame_cnt)
        print(patient_FP_frame_cnt)
        print(patient_TN_frame_cnt)
        print(patient_TP_frame_cnt)
        print(patient_TOTAL_frame_cnt)
        print('\n\n=============== \t\t ============= \n\n')

        print('Pateint Result Saved at \t ====> ', each_videoset_result_dir)

        # OOB_Metric
        patient_OOB_metric = calc_OOB_metric(patient_FN_frame_cnt, patient_FP_frame_cnt, patient_TN_frame_cnt, patient_TP_frame_cnt, patient_TOTAL_frame_cnt)

        # saving FN FP TP TN Metric
        patient_results_metric = {
        'Patient' : [videoset_name],
        'FP' : [patient_FP_frame_cnt],
        'TP' : [patient_TP_frame_cnt],
        'FN' : [patient_FN_frame_cnt],
        'TN' : [patient_TN_frame_cnt],
        'TOTAL' : [patient_TOTAL_frame_cnt],
        'GT_OOB' : [patient_TP_frame_cnt + patient_FN_frame_cnt],
        'GT_IB' : [patient_FP_frame_cnt + patient_TN_frame_cnt],
        'PREDICT_OOB' : [patient_FP_frame_cnt + patient_TP_frame_cnt],
        'PREDICT_IB' : [patient_FN_frame_cnt + patient_TN_frame_cnt],
        'GT_OOB_1FPS' : [patient_truth_oob_count],
        'GT_IB_1FPS' : [patient_truth_ib_count],
        'Confidence_Ratio' : [patient_OOB_metric]
        }

        # each metric per patient
        patient_result_metric_df = pd.DataFrame(patient_results_metric)

        print('Patient Metric Saved at \t ====> ', each_videoset_result_dir)
        patient_result_metric_df.to_csv(os.path.join(each_videoset_result_dir, 'Confidence_Ratio-{}-{}.csv'.format(args.mode, videoset_name)), mode="w")

        # append to total metric per patient
        patient_total_metric_df = pd.concat([patient_total_metric_df, patient_result_metric_df], ignore_index=True)

        # columns=['Patient', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'OOB_Metric'])
        print('')
        print(patient_total_metric_df)
        patient_total_metric_df.to_csv(os.path.join(results_save_dir, 'Patient_Total_metric-{}-{}.csv'.format(args.mode, os.path.basename(results_save_dir))), mode="w") # save on project direc
    
    print('\n\n=============== \t\t ============= \t\t ============= \n\n')


    
if __name__ == "__main__":
    ###  base setting for model testing ### 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    test_start()