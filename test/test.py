"""
For model test using pre-created test datset in tensor form.
"""

### for setting import mobule ###
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print(sys.path)

from OOB_RECOG.train.train_model import CAMIO

from OOB_RECOG.test.test_info_dict import make_data_sheet, load_data_sheet
from OOB_RECOG.test.test_info_dict import make_patients_aggregate_info, print_patients_info_yaml, patients_yaml_to_test_info_dict, sanity_check_info_dict
from OOB_RECOG.test.test_info_dict import convert_video_name_from_old_nas_policy
from OOB_RECOG.test.test_dataset import OOB_DB_Dataset, IDX_Sampler

from OOB_RECOG.evaluation.visual_gradcam import get_oob_grad_cam_from_video, get_oob_grad_cam_img, img_seq_to_gif, return_group

from OOB_RECOG.evaluation.visual_model import visual_metric_per_patients_ver2
### for setting import mobule ###

import cv2
from PIL import Image
import torch
import numpy as np
import pandas as pd
import matplotlib
import argparse
import time
import json
import datetime
import glob

import subprocess # for CLEAR PAGING CACHE

from pandas import DataFrame as df
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from torchvision import transforms
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from torch.utils.data import Dataset, DataLoader
import shutil

from natsort import natsorted, index_natsorted, order_by_index

# 21.06.10 HG 추가 - to load video for capture FP FN frame
from decord import VideoReader
from decord import cpu, gpu

from PIL import ImageFilter

from subprocess import Popen, PIPE

matplotlib.use('Agg')
EXCEPTION_NUM = -100 # full TN

LAPA_CASE = ['L_301', 'L_303', 'L_305', 'L_309', 'L_317', 'L_325', 'L_326', 'L_340', 'L_346', 'L_349', 'L_412', 'L_421', 'L_423', 'L_442', 'L_443',
                'L_450', 'L_458', 'L_465', 'L_491', 'L_493', 'L_496', 'L_507', 'L_522', 'L_534', 'L_535', 'L_550', 'L_553', 'L_586', 'L_595', 'L_605', 'L_607', 'L_625',
                'L_631', 'L_647', 'L_654', 'L_659', 'L_660', 'L_661', 'L_669', 'L_676', 'L_310', 'L_311', 'L_330', 'L_333', 'L_367', 'L_370', 'L_377', 'L_379', 'L_385',
                'L_387', 'L_389', 'L_391', 'L_393', 'L_400', 'L_402', 'L_406', 'L_408', 'L_413', 'L_414', 'L_415', 'L_418', 'L_419', 'L_427', 'L_428', 'L_430', 'L_433',
                'L_434', 'L_436', 'L_439', 'L_471', 'L_473', 'L_475', 'L_477', 'L_478', 'L_479', 'L_481', 'L_482', 'L_484', 'L_513', 'L_514', 'L_515', 'L_517', 'L_537',
                'L_539', 'L_542', 'L_543', 'L_545', 'L_546', 'L_556', 'L_558', 'L_560', 'L_563', 'L_565', 'L_568', 'L_569', 'L_572', 'L_574', 'L_575', 'L_577', 'L_580']

ROBOT_CASE = ['R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_10', 'R_13', 'R_14', 'R_15', 'R_17', 'R_18', 'R_19', 'R_22', 'R_48', 'R_56', 'R_74',
                'R_76', 'R_84', 'R_94', 'R_100', 'R_116', 'R_117', 'R_201', 'R_202', 'R_203', 'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_301',
                'R_302', 'R_303', 'R_304', 'R_305', 'R_310', 'R_311', 'R_312', 'R_313', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_336', 'R_338', 'R_339', 'R_340',
                'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357', 'R_358', 'R_362', 'R_363', 'R_369', 'R_372', 'R_376', 'R_378', 'R_379', 'R_386',
                'R_391', 'R_393', 'R_399', 'R_400', 'R_402', 'R_403', 'R_405', 'R_406', 'R_409', 'R_412', 'R_413', 'R_415', 'R_418', 'R_419', 'R_420', 'R_423', 'R_424',
                'R_427', 'R_436', 'R_445', 'R_449', 'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_526', 'R_532', 'R_533']

SUPPORT_MODEL = ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d',
                    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'squeezenet1_0', 'squeezenet1_1',
                    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, help='trained model_path')
parser.add_argument('--data_dir', type=str,
                    default='/data/ROBOT/Video', help='video_path :) ')
parser.add_argument('--anno_dir', type=str,
                    default='/data/OOB/V1_40', help='annotation_path :) ')
parser.add_argument('--data_sheet_dir', type=str,
                    default='./DATA_SHEET', help='datasheet_path, should include 3 file;[~/ANNOTATION_PATH_SHEET.yaml, DB_PATH_SHEET.yaml, VIDEO_PATH_SHEET.yaml] ')

parser.add_argument('--results_save_dir', type=str, help='inference results save path')

parser.add_argument('--mode', type=str, choices=['ROBOT', 'LAPA'], help='inference results save path')

## assets mode # 21.06.25 HG 추가 - VIDEO로 바로 Inferece (Inference 시간소요 up), test augmentation (30000, 3, 244, 244)로 미리 잘라논 INFERENCE TENSOR PICKLE DATA로 Inference (Inferece 시간소요 down)
parser.add_argument('--assets_mode', type=str, default='VIDEO', choices=['VIDEO', 'DB', 'TENSOR'], help='choose inferece assets VIDEO or DB or INFERENCE TENSOR(PICKLE DATA)')

## test model # 21.06.03 HG 수정 - Supported model [VGG]에 따른 choices 추가 # 21.06.05 HG 수정 [Squeezenet1_1] 추가 # 21.06.09 HG 추가 [EfficientNet Family]
parser.add_argument('--model', type=str,
                    choices=SUPPORT_MODEL, help='backbone model')

# inference frame step
parser.add_argument('--inference_step', type=int, default=5, help='inference frame step')

# inference video
parser.add_argument('--test_videos', type=str, nargs='+',
                    choices= LAPA_CASE + ROBOT_CASE,
                    help='inference video')

# infernece assets root path
parser.add_argument('--inference_assets_dir', type=str, help='inference assets root path')

args, _ = parser.parse_known_args()


# data transforms (output: tensor)
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
aug = data_transforms['test']



def npy_to_tensor(npy_img):
    pil_img = Image.fromarray(npy_img)
    img = aug(pil_img)
    
    return img

def batch_npy_to_tensor(batch_npy_img) : 
    b, h, w, c = batch_npy_img.shape
    return_tensor = torch.Tensor(b, 3, 224, 224) # batch, channel, height, width
    
    # loop for Batch size
    for b_idx in range(b) : 
        # convert pil
        pil_img = Image.fromarray(batch_npy_img[b_idx])
        
        # pil to tensor ('using test' aug)
        return_tensor[b_idx] = aug(pil_img)
    
    return return_tensor


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
    frame = int(idx % fps) # 21.06.10 HG 수정 - frame 소수점 truncate

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
# save FP FN frame from Video Record, 하나의 비디오에 대해 여러개 folder를 캡쳐하기 위해 사용
# LAPA, ROBOT 공용사용 가능
def save_video_frame_for_VR(video_path, frame_list_arr, save_path_arr, video_name, video_fps) : # using VideoRecord
    '''
    video_path = 'path'
    frame_list_arr = [[1st frame_list], [2nd frame_list]]
    frame_list_arr = ['1st save_path', '2nd save_path']
    video_name = 'video_name'
    video_fps = it's only used for idx to time
    '''

    print('TARGET VIDEO_PATH : ', video_path)
    video = VideoReader(video_path, ctx=cpu())
    
    for frame_list, save_path in zip(frame_list_arr, save_path_arr) : # multiple capture from one video
        print('TARGET FRAME : ', frame_list)

        for frame_idx in tqdm(frame_list, desc='Saving Frame From {} ... '.format(video_path)) : 
            video_frame = video[frame_idx].asnumpy()
            pil_image=Image.fromarray(video_frame)

            pil_image.save(fp=os.path.join(save_path, '{}-{:010d}.jpg'.format(video_name, frame_idx, idx_to_time(frame_idx, video_fps))))
            
    del video, video_frame
    print('======> DONE.')


# save video frame from frame_list | it will be saved in {save_path}/{video_name}-{frame_idx}.jpg
# ROBOT 사용가능, LAPA 불가
def save_video_frame_for_CV(video_path, frame_list, save_path, video_name, video_fps) : # using CV
    print('TARGET VIDEO_PATH : ', video_path)
    print('TARGET FRAME : ', frame_list)

    video = cv2.VideoCapture(video_path)

    for frame_idx in tqdm(frame_list, desc='Saving Frame From {} ... '.format(video_path)) :
        video_frame = video[i].asnumpy()
        
        video.set(1, frame_idx) # frame setting
        _, img = video.read() # read frame

        cv2.imwrite(os.path.join(save_path, '{}-{:010d}-{}.jpg'.format(video_name, frame_idx, idx_to_time(frame_idx, video_fps))), img)
    
    video.release()
    print('======> DONE.')

# 21.06.25 HG 추가 - VR Load시 객체 Memeory 누수현상을 줄이고자 사용
# save video frame from already loaded VideoRecoder 
# 이미 로드된 VideoRecoder를 parameter로 넘겨 해당 frame_list를 저장
def save_video_frame_for_loaded_VR(video, frame_list, save_path, video_name, video_fps) : # using loaded VR 
    '''
    video = VideoRecoder 객체
    frame_list = [idx_1, idx_2 ...]
    save_path = 'save_path'
    video_name = 'video_name'
    video_fps = it's only used for idx to time
    '''
    print('TARGET FRAME : ', frame_list)

    for frame_idx in tqdm(frame_list, desc='Saving Frame From {} to {}... '.format(video_name, save_path)) : 
        video_frame = video[frame_idx].asnumpy()
        pil_image=Image.fromarray(video_frame)

        pil_image.save(fp=os.path.join(save_path, '{}-{:010d}-{}.jpg'.format(video_name, frame_idx, idx_to_time(frame_idx, video_fps))))

# video index == 0 => DB idx == 0
def idx_to_DB_idx(idx):
    return idx

# DB_path : ~/L_301/01_G_01_L_301_xx0_01
def save_DB_frame(DB_path, frame_list, save_path, video_fps) : # for DB rule
 
    print('TARGET FRAME : ', frame_list)
    
    video_name = os.path.basename(DB_path) # parents name (01_G_01_L_301_xx0_01)
    cnt = 0

    img_list = glob.glob(os.path.join(DB_path, '*.jpg')) # ALL img into DB path
    
    for target_idx in tqdm(frame_list, desc='Saving Frame From {} to {}...'.format(video_name, save_path)): 
        target_img = '{}-{:010d}.jpg'.format(video_name, idx_to_DB_idx(target_idx)) # 01_G_01_L_301_xx0_01-0000000001.jpg
        target_path = os.path.join(DB_path, target_img)
        
        if target_path in img_list :
            save_file_name = '{}-{:010d}-{}.jpg'.format(video_name, target_idx, idx_to_time(target_idx, video_fps))
            
            # PIL LOAD SAVE
            '''
            pil_img = Image.open(target_path)     
            pil_img.save(fp=os.path.join(save_path, save_file_name))
            '''

            # COPY
            shutil.copy(target_path, os.path.join(save_path, save_file_name))

            cnt+=1
    
    
    print('SAVE DB FRAME : SUCCESS {} | REQUEST {}'.format(cnt, len(frame_list)))
            
    



# 21.06.25 HG 추가 - GRADCAM for FP Frame => call function in visual_gradcam.py [return_group, get_oob_grad_cam_from_video, img_seq_to_gif]
def save_fp_frame_gradcam(model_path, model_name, data_sheet_dir, consensus_results_path, inference_img_dir_dict, save_dir, title_name, assets_mode) :
    assert assets_mode in ['VIDEO', 'DB'], 'NO SUPPORT ASSETS MODE [GRADCAM] INPUT ASSETS MODE : {}'.format(assets_mode)
    # video_dir의 모든 video parsing
    '''
    all_video_path = []
    video_ext_list = ['mp4', 'MP4', 'mpg']

    for ext in video_ext_list :
        all_video_path.extend(glob.glob(video_dir +'/*.{}'.format(ext)))
    '''

    # load DATA SHEET
    VIDEO_PATH_SHEET, ANNOTATION_PATH_SHEET, DB_PATH_SHEET = load_data_sheet(data_sheet_dir)


    # consensus_results csv loading
    consensus_results_df = pd.read_csv(consensus_results_path)
    


    # calc FP frame
    metric_frame_dict = return_metric_frame(consensus_results_df)
    FP_consensus_df = metric_frame_dict['FP_df']

    # group by 'video_name'
    FP_df_per_group = return_group(FP_consensus_df)
    
    # video_name별로 FP Frame Gradcam
    for video_name, FP_df in FP_df_per_group.items() :
        print('video_name : ', video_name)

        # video_dir 찾기
        # video_path_list = [v_path for v_path in all_video_path if video_name in v_path]
        video_sheet_key = '_'.join(video_name.split('_')[-4:]) # 01_G_01_R_1_ch1_11
        video_path = [VIDEO_PATH_SHEET.get(video_sheet_key, '')]

        # for DB, set inference img path 
        inference_img_dir = inference_img_dir_dict[video_name]

        # non video_dir 
        if video_path != '' : # has video_dir
            # video_path = video_path_list[0]

            # set save dir and make dir
            # each_save_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(video_path))[0])
            each_save_dir = os.path.join(save_dir, video_name)

            try:
                if not os.path.exists(each_save_dir) :
                        os.makedirs(each_save_dir)
            except OSError :
                print('ERROR : Creating Directory, ' + each_save_dir)

            # gradcam visual save in each save_dir
            if assets_mode == 'VIDEO':
                try :
                    get_oob_grad_cam_from_video(model_path, model_name, video_path, FP_df, each_save_dir, title_name)
                except:
                    pass
            
            elif assets_mode == 'DB':
                try :
                    get_oob_grad_cam_img(model_path, model_name, inference_img_dir, each_save_dir, title_name)
                except :
                    pass

            # save_dir img to gif
            print('\n\n===> CONVERTING GIF\n\n')
            all_results_img_path = natsorted(glob.glob(each_save_dir +'/*{}'.format('jpg'))) # 위에서 저장한 img 모두 parsing
            img_seq_to_gif(all_results_img_path, os.path.join(each_save_dir, '{}-GRADCAM.gif'.format(video_name))) # seqence 이므로 sort 하여 append
            print('\n\n===> DONE\n\n')
    
        else : # no video_dir
            assert False, "ERROR : Video No Exist"



# HG 추가 - 기존 calc_OOB_metric function 대체
# calc OOB Evaluation Metric
def calc_OOB_Evaluation_metric(FN_cnt, FP_cnt, TN_cnt, TP_cnt) :
	base_denominator = FP_cnt + TP_cnt + FN_cnt	
	# init
	EVAL_metric = {
		'CONFIDENCE_metric' : EXCEPTION_NUM,
		'correspondence' : EXCEPTION_NUM,
		'UN_correspondence' : EXCEPTION_NUM,
		'OVER_estimation' : EXCEPTION_NUM,
		'UNDER_estimtation' : EXCEPTION_NUM,
		'FN' : FN_cnt,
		'FP' : FP_cnt,
		'TN' : TN_cnt,
		'TP' : TP_cnt,
		'TOTAL' : FN_cnt + FP_cnt + TN_cnt + TP_cnt
	}


	if base_denominator > 0 : # zero devision except check, FN == full
		EVAL_metric['CONFIDENCE_metric'] = (TP_cnt - FP_cnt) / base_denominator
		EVAL_metric['correspondence'] = TP_cnt /  base_denominator
		EVAL_metric['UN_correspondence'] = (FP_cnt + FN_cnt) /  base_denominator
		EVAL_metric['OVER_estimation'] = FP_cnt / base_denominator
		EVAL_metric['UNDER_estimtation'] = FN_cnt / base_denominator
	
	return EVAL_metric

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


# parsing video info from ffmpeg
def get_video_fps(video_path):
    cmds_list = []

    fps = EXCEPTION_NUM
    cmd = ['ffmpeg', '-i', video_path, '2>&1', '|', 'sed', '-n', '"s/.*, \(.*\) fp.*/'+'\\'+'1/p"']
    # sed -n "s/.*, \(.*\) fp.*/\1/p"
    # cmd = ['ffprobe', '-v', 'error', '-select_streams v:0', '-count_packets', '-show_entries stream=nb_read_packets', '-of', 'csv=p=0', video_path]
    cmd = ' '.join(cmd)
    cmds_list.append([cmd])
    
    print('GET VIDEO FPS USIGN FFMPEG : {}'.format(cmd), end= ' ')
    
    # procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE  , shell=True) for cmd in cmds_list]
    procs_list = [Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True) for cmd in cmds_list]

    for proc in procs_list:
        print('ing ...')
        # proc.wait() # communicate() # 자식 프로세스들이 I/O를 마치고 종료하기를 기다림
        out = proc.communicate()
        print("Processes are done")

    fps = out[0].decode('UTF-8').rstrip() # byte to str
    fps = float(fps)
    return fps


def get_video_length(video_path):
    cmds_list = []

    video_len = EXCEPTION_NUM

    cmd = ['ffprobe', '-v', 'error', '-select_streams v:0', '-count_packets', '-show_entries stream=nb_read_packets', '-of', 'csv=p=0', video_path]
    cmd = ' '.join(cmd)
    cmds_list.append([cmd])
    
    print('GET VIDEO LENGTH USIGN FFPROBE : {}'.format(cmd), end= ' ')
    
    # procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE  , shell=True) for cmd in cmds_list]
    procs_list = [Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True) for cmd in cmds_list]

    for proc in procs_list:
        print('ing ...')
        # proc.wait() # communicate() # 자식 프로세스들이 I/O를 마치고 종료하기를 기다림
        out = proc.communicate()
        print("Processes are done")

    video_len = out[0].decode('UTF-8').rstrip() # byte to str
    video_len = int(video_len)
    return video_len




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
    '''
    if args.mode == 'LAPA' : 
        # starting inference
        test_for_lapa(args.data_dir, args.anno_dir, args.inference_assets_dir, args.results_save_dir, model, args.test_videos, args.inference_step)

    else: # robot
        # starting inference
        test_for_robot(args.data_dir, args.anno_dir, args.inference_assets_dir, args.results_save_dir, model, args.test_videos, args.inference_step)
    '''

    # make patinets aggregation file from DATA SHEET and convert to info_dict
    patinets_info_save_path = os.path.join(args.results_save_dir, 'patiensts_info.yaml') # save path of patinets aggregation file (.yaml)
    data_sheet_dir = args.data_sheet_dir
    info_dict = prepare_test_info_dict(args.test_videos, data_sheet_dir, patinets_info_save_path) # load Data sheet and make patinets aggregation file, then convert to info_dict
    
    # test
    test(info_dict, model, args.results_save_dir, args.inference_step)

    # finish time stamp
    finishTime = time.time()
    f_tm = time.localtime(finishTime)

    log_txt = '\n\nFINISHED AT : \t' + time.strftime('%Y-%m-%d %I:%M:%S %p \n', f_tm)
    save_log(log_txt, os.path.join(args.results_save_dir, 'log.txt')) # save log

'''
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
    # fps = 30 # 21.06.10 HG 수정 - Not used

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
    test(info_dict, model, results_save_dir, inference_step) # 21.06.10 HG 수정 - automatic FPS setting for each video FPS

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
    # fps = 30 # 21.06.10 HG 수정 - Not used

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
    test(info_dict, model, results_save_dir, inference_step) # 21.06.10 HG 수정 - automatic FPS setting for each video FPS
'''

# load Data sheet and make patinets aggregation file, then convert to info_dict
def prepare_test_info_dict(patient_list, data_sheet_dir, patinets_info_save_path):
    # make DATA SHEET
    make_data_sheet(data_sheet_dir)

    # load DATA SHEET
    VIDEO_PATH_SHEET, ANNOTATION_PATH_SHEET, DB_PATH_SHEET = load_data_sheet(data_sheet_dir)
    
    # make patinets aggregation file
    make_patients_aggregate_info(patient_list, VIDEO_PATH_SHEET, ANNOTATION_PATH_SHEET, DB_PATH_SHEET, patinets_info_save_path)

    # print patinets aggregation file
    print_patients_info_yaml(patinets_info_save_path)

    # convert patients aggregation file to info_dict
    info_dict = patients_yaml_to_test_info_dict(patinets_info_save_path)

    # sanity check and modify info_dict
    info_dict = sanity_check_info_dict(info_dict)

    return info_dict

# project_name is only for use for title in total_metric_df.csv
def test(info_dict, model, results_save_dir, inference_step) : # 21.06.10 HG 수정 - automatic FPS setting for each video FPS
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
    total_metric_df = pd.DataFrame(index=range(0, 0), columns=['Video_set', 'Video_name', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'Confidence_Ratio', 'Over_Ratio', 'Under_Ratio', 'Correspondence', 'Un_Correspondence']) # row cnt is same as checking vidoes length
    patient_total_metric_df = pd.DataFrame(index=range(0, 0), columns=['Patient', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'Confidence_Ratio', 'Over_Ratio', 'Under_Ratio', 'Correspondence', 'Un_Correspondence']) # row cnt is same as total_videoset_cnt

    # loop from total_videoset_cnt
    for i, (video_path_list, anno_info_list, DB_path_list, infernece_assets_path_list) in enumerate(zip(info_dict['video'], info_dict['anno'], info_dict['DB'], info_dict['inference_assets']), 1):
        ##### PARSING VIDEOSET NAME #####
        if args.mode == 'ROBOT': # R000001, ch1_video_01.mp4
            ### EXCEPTION RULE
            EXCEPTION_RULE = {'ch1_video_01_6915320_RDG.mp4': 'R_76_ch1_01',
                       'ch1_video_01_8459178_robotic\ subtotal.mp4': 'R_84_ch1_01',
                       '01_G_01_R_391_ch2_06.mp4': 'R_391_ch2_06'}

            if os.path.basename(video_path_list[0]) in EXCEPTION_RULE:
                new_nas_policy_name = EXCEPTION_RULE.get(video_path_list[0], '-')
            else: 
                new_nas_policy_name = convert_video_name_from_old_nas_policy(video_path_list[0]) # R_1_ch1_01
            
            op_method, patient_idx, video_channel, video_slice = new_nas_policy_name.split('_')
        
        elif args.mode == 'LAPA':
            hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice = os.path.splitext(os.path.basename(video_path_list[0]))[0].split('_') # parsing videoset name

        videoset_name = '{}_{}'.format(op_method, patient_idx)
        ##### ##### ##### ##### ##### #####

        # init for patient results_dict
        patient_video_list = []
        patient_frame_idx_list = []
        patient_time_list = []
        patient_gt_list = []
        patient_target_img_list = [] # for DB
        patient_predict_list = []
        patient_truth_oob_count = 0
        patient_truth_ib_count = 0
        
        # for DB gradcam
        fp_frame_saved_dir_dict = {}

        # create base folder for save results each video set
        each_videoset_result_dir = os.path.join(results_save_dir, videoset_name) # '~~~/results/R022' , '~~~/results/R011' ..
        try :
            if not os.path.exists(os.path.join(each_videoset_result_dir)) :
                os.makedirs(each_videoset_result_dir)
        except OSError :
            print('ERROR : Creating Directory, ' + each_videoset_result_dir)


        print('COUNT OF VIDEO SET | {} / {} \t\t ======>  VIDEO SET | {}'.format(i, total_videoset_cnt, videoset_name))
        print('NUMBER OF VIDEO : {} | NUMBER OF ANNOTATION INFO : {} | NUMBER OF DB_PATH : {}'.format(len(video_path_list), len(anno_info_list), len(DB_path_list)))
        print('NUMBER OF INFERNECE ASSETS {} \n==> ASSETS LIST : {}'.format(len(infernece_assets_path_list), infernece_assets_path_list))
        print('RESULTS SAVED AT \t\t\t ======>  {}'.format(each_videoset_result_dir))
        print('\n')
             
        
        # extract info for each video_path
        for video_path, anno_info, DB_path, each_video_infernece_assets_path_list in zip(video_path_list, anno_info_list, DB_path_list, infernece_assets_path_list) :
            ##### PARSING VIDEO NAME #####

            video_file_name = os.path.basename(video_path)

            if args.mode == 'ROBOT': # R000001, ch1_video_01.mp4
                ### EXCEPTION RULE
                EXCEPTION_RULE = {'ch1_video_01_6915320_RDB.mp4': 'R_76_ch1_01',
                        'ch1_video_01_8459178_robotic subtotal.mp4': 'R_84_ch1_01',
                        '01_G_01_R_391_ch2_06.mp4': 'R_391_ch2_06'}

                if video_file_name in EXCEPTION_RULE:
                    new_nas_policy_name = EXCEPTION_RULE[video_file_name]
                else: 
                    new_nas_policy_name = convert_video_name_from_old_nas_policy(video_path) # ~/R00001/R_1_ch1_01
                
                video_name = '01_G_01_'+ new_nas_policy_name # 01_G_01_R_1_ch1_01
            
            elif args.mode == 'LAPA':
                video_name = os.path.splitext(video_file_name)[0] # only video name

            ##### ##### ##### ##### ##### #####

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
            '''            
            video = cv2.VideoCapture(video_path)
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # it's okay to use CV
            video_fps = video.get(cv2.CAP_PROP_FPS) # it's okay to use CV
            
            video.release()
            del video
            '''
            
            # for DB gradcam
            fp_frame_saved_dir_dict[video_name] = fp_frame_saved_dir

            # using VR(len) & CV (fps) # 21.06.10 HG Change to VideoRecoder
            '''
            video_cap = cv2.VideoCapture(video_path)
            video_fps = video_cap.get(cv2.CAP_PROP_FPS)

            video_cap.release()
            del video_cap

            video = VideoReader(video_path, ctx=cpu())
            video_len = len(video)
            # del video, video_cap
            '''

            # using ffmpeg
            video_len = get_video_length(video_path)
            video_fps = get_video_fps(video_path)

            print('\tTarget video : {} | Total Frame : {} | Video FPS : {} '.format(video_name, video_len, video_fps))
            print('\tAnnotation Info : {}'.format(anno_info))

            ### check idx -> time
            if anno_info : # event 
                for start, end in anno_info :
                    print([idx_to_time(start, video_fps), idx_to_time(end, video_fps)])
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
            target_img_list = [] # target img path from DB # only for using in args.mode=DB

            FP_frame_cnt = 0
            FN_frame_cnt = 0
            TP_frame_cnt = 0
            TN_frame_cnt = 0
            TOTAL_frame_cnt = 0
            # frame_check_cnt = 0 # loop cnt

            # temp_cnt = 0 ### dummy for test
            # for video inference
            # batch slice from infernece_frame_idx
            BATCH_SIZE = 256
            
            
            with torch.no_grad() : # autograd 끔 - 메모리 사용량 줄이고, 연산 속도 높힘. 사실상 안 쓸 gradient 라서 inference 시에 굳이 계산할 필요 없음. 
                model.eval() # layer 의 동작을 inference(eval) mode로 변경, 모델링 시 training과 inference 시에 다르게 동작하는 layer 들이 존재함. (e.g. Dropout layer, BatchNorm)

                ######### ######### ######### #########
                ########## FOR USING VIDEO #########
                if args.assets_mode == 'VIDEO' :
                    print('\n\t########## FOR USING VIDEO #########\n')
                    TOTAL_INFERENCE_FRAME_INDICES = list(range(0, video_len, inference_step))
                
                    start_pos = 0
                    end_pos = len(TOTAL_INFERENCE_FRAME_INDICES)

                    print('TOTAL_INFERENCE_FRAME_INDICES')
                    print(TOTAL_INFERENCE_FRAME_INDICES)
                    
                    for idx in tqdm(range(start_pos, end_pos + BATCH_SIZE, BATCH_SIZE),  desc='Inferencing... \t ==> {}'.format(video_name)):
                        FRAME_INDICES = TOTAL_INFERENCE_FRAME_INDICES[start_pos:start_pos + BATCH_SIZE] # batch video frame idx
                        if FRAME_INDICES != []:
                            BATCH_INPUT = video.get_batch(FRAME_INDICES).asnumpy() # get batch (batch, height, width, channel)

                            # convert batch tensor
                            BATCH_TENSOR = batch_npy_to_tensor(BATCH_INPUT).cuda()

                            # inferencing model
                            BATCH_OUTPUT = model(BATCH_TENSOR)

                            # predict
                            BATCH_PREDICT = torch.argmax(BATCH_OUTPUT.cpu(), 1)
                            BATCH_PREDICT = BATCH_PREDICT.tolist()

                            # save results
                            frame_idx_list+=FRAME_INDICES
                            predict_list+= list(BATCH_PREDICT)

                            gt_list+=(lambda in_list, indices_list : [in_list[i] for i in indices_list])(truth_list, FRAME_INDICES) # get in_list elements from indices index 
                            time_list+=[idx_to_time(frame_idx, video_fps) for frame_idx in FRAME_INDICES] # FRAME INDICES -> time

                            target_img_list+= '-' * len(FRAME_INDICES)

                            # print('FRAME_INDICES :', FRAME_INDICES)
                            # print('BATCH INPUT TENSOR SIZE : {}'.format(BATCH_TENSOR.size))
                            # print('frame_idx_list : {} \n predict_list : {} \n gt_list : {}'.format(frame_idx_list, predict_list, gt_list))
                            # print('\n\n')

                        start_pos = start_pos + BATCH_SIZE

                ########## FOR USING VIDEO #########
                ######### ######### ######### #########
                
                # ---------------------------------- #
                ######### ######### ######### #########
                ########## FOR USING DB #########  
                elif args.assets_mode == 'DB' :
                    print('\n\t########## FOR USING DB #########\n')

                    # DB dataset
                    print('VIDEO :', video_path)
                    print('DB PATH : ', DB_path)
                    
                    test_dataset = OOB_DB_Dataset(DB_path)

                    # target idx
                    TARGET_IDX_LIST = list(range(0, len(test_dataset), inference_step))

                    # check video len with DB dataset
                    try : 
                        log_txt='\nVIDEO NAME : {} \t CHECK LENGTH | video_len : {} | DB_len : {} '.format(video_name, video_len, len(test_dataset))
                        print(log_txt)
                        save_log(log_txt, os.path.join(results_save_dir, 'log.txt')) # save log
                        assert video_len >= len(test_dataset), 'NOT COMPATIABLE(OVER) LENGTH | video_len : {} | DB_len : {}'.format(video_len, len(test_dataset))
                    except :
                        log_txt='\t===> NOT COMPATIABLE(OVER)'
                        print('NOT COMPATIABLE(OVER) LENGTH | video_len : {} | DB_len : {}'.format(video_len, len(test_dataset)))
                        save_log(log_txt, os.path.join(results_save_dir, 'log.txt')) # save log
                        TARGET_IDX_LIST = list(range(0, video_len, inference_step)) # because of truth_list index (out of index)
                        pass

                    # Set index for batch
                    s = IDX_Sampler(TARGET_IDX_LIST, batch_size=BATCH_SIZE)

                    # set dataloader with custom batch sampler
                    dl = DataLoader(test_dataset, batch_sampler=list(s))
                    
                    # inferencing model
                    for sample in tqdm(dl, desc='Inferencing... \t ==> {} | {}'.format(video_name, DB_path)) :
                        BATCH_INPUT = sample['img'].cuda()
                        BATCH_OUTPUT = model(BATCH_INPUT)

                        # predict
                        BATCH_PREDICT = torch.argmax(BATCH_OUTPUT.cpu(), 1)
                        BATCH_PREDICT = BATCH_PREDICT.tolist()

                        # save results
                        predict_list+= list(BATCH_PREDICT)
                        target_img_list+=sample['img_path'] # target img path

                        


                    
                    frame_idx_list += TARGET_IDX_LIST
                    gt_list+=(lambda in_list, indices_list : [in_list[i] for i in indices_list])(truth_list, TARGET_IDX_LIST) # get in_list elements from indices index 
                    time_list+=[idx_to_time(frame_idx, video_fps) for frame_idx in TARGET_IDX_LIST] # FRAME INDICES -> time
                   
                

                ######### ######### ######### #########
                ########## FOR USING TENSOR #########  
                elif args.assets_mode == 'TENSOR' :
                    print('\n\t########## FOR USING TENSOR #########\n')
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
                                time_list+=[idx_to_time(frame_idx, video_fps) for frame_idx in FRAME_INDICES] # FRAME INDICES -> time
                                
                                target_img_list+= '-' * len(FRAME_INDICES)


                            start_pos = start_pos + BATCH_SIZE
                        

                        del infernce_asset # free memory
                

                ########## FOR USING TENSOR #########
                ######### ######### ######### #########

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
                'predict' : predict_list,
                'target_img_list': target_img_list # for DB
            }

            # append inference results for patient
            patient_video_list.append(video_name)
            patient_frame_idx_list.append(frame_idx_list)
            patient_time_list.append(time_list)
            patient_gt_list.append(gt_list)
            patient_predict_list.append(predict_list)
            patient_target_img_list.append(target_img_list) # for DB
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
            # FP & FN frame
            print('\n\n=============== \tSAVE FP & FN frame \t ============= \n\n')
            #save_video_frame_for_VR(video_path, [list(metric_frame['FP_df']['frame']), list(metric_frame['FN_df']['frame'])], [fp_frame_saved_dir, fn_frame_saved_dir], video_name)
            if args.assets_mode == 'VIDEO': 
                save_video_frame_for_loaded_VR(video, list(metric_frame['FP_df']['frame']), fp_frame_saved_dir, video_name, video_fps) # Saving FP
                save_video_frame_for_loaded_VR(video, list(metric_frame['FN_df']['frame']), fn_frame_saved_dir, video_name, video_fps) # Saving FN

            # for DB
            elif args.assets_mode == 'DB':
                save_DB_frame(DB_path, list(metric_frame['FP_df']['frame']), fp_frame_saved_dir, video_fps) # Saving FP
                save_DB_frame(DB_path, list(metric_frame['FN_df']['frame']), fn_frame_saved_dir, video_fps) # Saving FN

            # del video # delete Video

            # Evalutation Metric per VIDEO
            # CONFIDENCE_metric | correspondence | UN_correspondence | OVER_estimation | UNDER_estimtation | FN | FP | TN | TP | TOTAL
            # inital value
            video_evaluation_dict, video_confidence_metric, video_over_metric, video_under_metric, video_correspondence, video_uncorrespondence = None, None, None, None, None, None

            video_evaluation_dict = calc_OOB_Evaluation_metric(FN_frame_cnt, FP_frame_cnt, TN_frame_cnt, TP_frame_cnt)
            video_confidence_metric = video_evaluation_dict['CONFIDENCE_metric']
            video_over_metric = video_evaluation_dict['OVER_estimation']
            video_under_metric = video_evaluation_dict['UNDER_estimtation']
            video_correspondence = video_evaluation_dict['correspondence']
            video_uncorrespondence = video_evaluation_dict['UN_correspondence']

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
                'Confidence_Ratio' : [video_confidence_metric],
                'Over_Ratio' : [video_over_metric],
                'Under_Ratio' : [video_under_metric],
                'Correspondence' : [video_correspondence],
                'Un_Correspondence' : [video_uncorrespondence]
            }

            # each metric per video
            result_metric_df = pd.DataFrame(results_metric)
            
            # save
            print('Metric Saved at \t ====> ', each_video_result_dir)
            result_metric_df.to_csv(os.path.join(each_video_result_dir, 'Confidence_Ratio-{}-{}.csv'.format(args.mode, video_name)), mode="w")

            # append metric
            # columns=['Video_set', 'Video_name', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'Confidence_Ratio', 'Over_Ratio', 'Under_Ratio', Correspondence, Un_Correspondence])           
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
            plt.suptitle('OOB_CLASS [{}] | IB_CLASS [{}] | FP : {} | TP : {} | FN : {} | TN : {} | TOTAL : {} | Confidence_Ratio : {} '.format(OOB_CLASS, IB_CLASS, FP_frame_cnt, TP_frame_cnt, FN_frame_cnt, TN_frame_cnt, TOTAL_frame_cnt, video_confidence_metric)); # HG 수정 - 21.06.27 변수이름 변경에 따른 변경
            plt.ylabel('class'); plt.xlabel('frame');
            plt.legend(loc='center right');

            plt.savefig(os.path.join(each_video_result_dir, 'plot_{}.png'.format(video_name)));

            # save confusion matrix
            # def calc_confusion_matrix(inference_results_df['truth'], inference_results_df['predict'])
            saved_text = calc_confusion_matrix(gt_list, predict_list)
            saved_text += '\n\nFP\t\tTP\t\tFN\t\tTN\t\tTOTAL\n'
            saved_text += '{}\t\t{}\t\t{}\t\t{}\t\t{}\n\n'.format(FP_frame_cnt, TP_frame_cnt, FN_frame_cnt, TN_frame_cnt, TOTAL_frame_cnt)
            saved_text += 'Confidence_Ratio : {:.4f}'.format(video_confidence_metric) # HG 수정 - 21.06.27 변수이름 변경에 따른 변경

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
        for patient_video, patient_frame_idx, patient_time, patient_gt, patient_predict, patient_target_img in zip(patient_video_list, patient_frame_idx_list, patient_time_list, patient_gt_list, patient_predict_list, patient_target_img_list) :
            # saving inferece result per patient
            temp_patient_result_dict = {
                'video_name' : [patient_video]*len(patient_gt),
                'frame' : patient_frame_idx,
                'time' : patient_time,
                'truth' : patient_gt,
                'predict' : patient_predict,
                'target_img': patient_target_img,
            }

            # re-index time and frame_idx
            patient_inference_results_df = pd.concat([patient_inference_results_df, df(temp_patient_result_dict)], ignore_index=True)

        # re-index consensus_frame
        patient_inference_results_df['consensus_frame'] = [frame * inference_step for frame in range(len(patient_inference_results_df))]
        # calc consensus time
        patient_inference_results_df['consensus_time'] = patient_inference_results_df.apply(lambda x : idx_to_time(x['consensus_frame'], video_fps), axis=1)

        # save
        print('Patient Result Saved at \t ====> ', each_videoset_result_dir)
        patient_inference_results_df_save_path = os.path.join(each_videoset_result_dir, 'Inference-{}-{}.csv'.format(args.mode, videoset_name))
        patient_inference_results_df.to_csv(patient_inference_results_df_save_path, mode="w")

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

        print('Patient Result Saved at \t ====> ', each_videoset_result_dir)

        # OOB_Metric
        # Evalutation Metric per PATIENT
        # CONFIDENCE_metric | correspondence | UN_correspondence | OVER_estimation | UNDER_estimtation | FN | FP | TN | TP | TOTAL
        # inital value
        patient_evaluation_dict, patient_confidence_metric, patient_over_metric, patient_under_metric, patient_correspondence, patient_uncorrespondence = None, None, None, None, None, None
        patient_evaluation_dict = calc_OOB_Evaluation_metric(patient_FN_frame_cnt, patient_FP_frame_cnt, patient_TN_frame_cnt, patient_TP_frame_cnt)
        patient_confidence_metric = patient_evaluation_dict['CONFIDENCE_metric']
        patient_over_metric = patient_evaluation_dict['OVER_estimation']
        patient_under_metric = patient_evaluation_dict['UNDER_estimtation']
        patient_correspondence = patient_evaluation_dict['correspondence']
        patient_uncorrespondence = patient_evaluation_dict['UN_correspondence']

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
        'Confidence_Ratio' : [patient_confidence_metric],
        'Over_Ratio' : [patient_over_metric],
        'Under_Ratio' : [patient_under_metric],
        'Correspondence' : [patient_correspondence],
        'Un_Correspondence' : [patient_uncorrespondence]
        }

        # each metric per patient
        patient_result_metric_df = pd.DataFrame(patient_results_metric)

        print('Patient Metric Saved at \t ====> ', each_videoset_result_dir)
        patient_result_metric_df.to_csv(os.path.join(each_videoset_result_dir, 'Confidence_Ratio-{}-{}.csv'.format(args.mode, videoset_name)), mode="w")

        # append to total metric per patient
        patient_total_metric_df = pd.concat([patient_total_metric_df, patient_result_metric_df], ignore_index=True)

        # columns=['Patient', 'FP', 'TP', 'FN', 'TN', 'TOTAL', 'GT_OOB', 'GT_IB', 'PREDICT_OOB', 'PREDICT_IB', 'GT_OOB_1FPS', 'GT_IB_1FPS', 'Confidence_Ratio', Over_Ratio, Under_Ratio, Correspondence, Un_Correspondence])
        print('')
        print(patient_total_metric_df)
        patient_total_metric_df.to_csv(os.path.join(results_save_dir, 'Patient_Total_metric-{}-{}.csv'.format(args.mode, os.path.basename(results_save_dir))), mode="w") # save on project direc

        ############################
        ######  VISUAL_FRAME  ######
        visual_frame_save_dir = os.path.join(each_videoset_result_dir, 'VISUAL')

        try :
            if not os.path.exists(visual_frame_save_dir) :
                os.makedirs(visual_frame_save_dir)
        except OSError :
            print('ERROR : Creating Directory, ' + visual_frame_save_dir)

        visual_sh_command = \
        "python visual_frame.py " +\
        "--title_name " + '"{}-INFERENCE STEP_{}" '.format(args.model, inference_step) +\
        "--sub_title_name " + '"{}" '.format(videoset_name) +\
        "--GT_path " + '"{}" '.format(patient_inference_results_df_save_path) +\
        "--model_name " + '"{}" '.format(args.model) +\
        "--model_infernce_path " + '"{}" '.format(patient_inference_results_df_save_path) +\
        "--results_save_dir " + '"{}" '.format(visual_frame_save_dir) +\
        "--INFERENCE_STEP " + '{} '.format(inference_step) +\
        "--WINDOW_SIZE " + '{} '.format(1000) +\
        "--OVERLAP_SECTION_NUM " + '{} '.format(2)

        print(visual_sh_command)
        
        subprocess.run(visual_sh_command, shell=True)

        ######  VISUAL_FRAME  ######
        ############################


        #######################
        ######  GRADCAM  ######
        # FP Frame GRADCAM saved folder for each video
        print('\n\n=============== \t\t GRADCAM PROCESSING \t\t ============= \n\n')
        fp_frame_gradcam_saved_dir = os.path.join(each_videoset_result_dir, 'GRADCAM', 'FP') # '~~~/results/R022/GRADCAM/FP'
        fp_gradcam_title = 'FP | {} | {}'.format(args.model, videoset_name) # 'FP | mobilenet_v3_large | R_1'

        try :
            if not os.path.exists(fp_frame_gradcam_saved_dir) :
                os.makedirs(fp_frame_gradcam_saved_dir)
        except OSError :
            print('ERROR : Creating Directory, ' + fp_frame_gradcam_saved_dir)
        
        print('GRADCAM FP Result Saved at \t ====> ', fp_frame_gradcam_saved_dir)
        # def save_fp_frame_gradcam(model_path, model_name, video_dir, consensus_results_path, inference_dir_dict, save_dir, title_name, assets_mode)
        # fp_frame_saved_dir_dict is only use for DB
        
        save_fp_frame_gradcam(args.model_path, args.model, args.data_sheet_dir, patient_inference_results_df_save_path, fp_frame_saved_dir_dict,fp_frame_gradcam_saved_dir, fp_gradcam_title, args.assets_mode) # GRADCAM to sequcence gif

        ######  GRADCAM  ######
        #######################

        # clear Paging Cache because of VideoRecoder I/O CACHE [docker run -it --name cam_io_hyeongyu -v /proc:/writable_proc -v /home/hyeongyuc/code/OOB_Recog:/OOB_RECOG -v /nas/OOB_Project:/data -p 6006:6006  --gpus all --ipc=host oob:1.0]
        print('\n\n\t ====> CLEAN PAGINGCACHE, DENTRIES, INODES "echo 1 > /writable_proc/sys/vm/drop_caches"\n\n')
        subprocess.run('sync', shell=True)
        subprocess.run('echo 1 > /writable_proc/sys/vm/drop_caches', shell=True) ### For use this Command you should make writable proc file when you run docker
    
    print('\n\n=============== \t\t ============= \t\t ============= \n\n')
    
    ###########################################
    ######  METRIC VISUAL PER PATIENTS  ######
    metric_visual_csv_path = os.path.join(results_save_dir, 'Patient_Total_metric-{}-{}.csv'.format(args.mode, os.path.basename(results_save_dir)))
    metric_visual_results_path = os.path.join(results_save_dir, 'Patient_Total_metric-{}-{}.png'.format(args.mode, os.path.basename(results_save_dir)))
    metric_visual_title = 'Metric per Patients ({})'.format(args.model)
    visual_metric_per_patients_ver2(metric_visual_csv_path, metric_visual_title, metric_visual_results_path)
    ######  METRIC VISUAL PER PATIENTS  ######
    ###########################################

    #####################################
    ######  FOR VISUAL ASSETS CSV  ######
    metric_pivot_save_dir = os.path.join(results_save_dir, 'PIVOT')
    
    try :
        if not os.path.exists(metric_pivot_save_dir) :
            os.makedirs(metric_pivot_save_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + metric_pivot_save_dir)
    
    # CR Pivot
    convert_to_visual_assets(metric_visual_csv_path, args.model, 'Confidence_Ratio', os.path.join(metric_pivot_save_dir, 'Patinet_CR-{}-{}.csv'.format(args.mode, os.path.basename(results_save_dir))))

    # OR Pivot
    convert_to_visual_assets(metric_visual_csv_path, args.model, 'Over_Ratio', os.path.join(metric_pivot_save_dir, 'Patinet_OR-{}-{}.csv'.format(args.mode, os.path.basename(results_save_dir))))
    ######  FOR VISUAL ASSETS CSV  ######
    #####################################

    print('\n\n=============== \t\t ============= \t\t ============= \n\n')


def convert_to_visual_assets(patient_total_metric_csv_path, model_name, value_col_name, save_path):

    assert value_col_name in ['Confidence_Ratio', 'Over_Ratio', 'Under_Ratio', 'Correspondence', 'Un_Correspondence'], 'NOT SOPPORT VALUE'
    
    # 0. load csv and set init val
    patient_total_metric_df = pd.read_csv(patient_total_metric_csv_path)

    #### Confidence
    # 1. sorting by patinets
    target_col_name = 'Patient'
    patient_total_metric_df['Model'] = model_name
    
    patient_total_metric_df = patient_total_metric_df.reindex(index=order_by_index(patient_total_metric_df.index, index_natsorted(patient_total_metric_df[target_col_name])))

    print(patient_total_metric_df)

    convert_df = patient_total_metric_df.pivot(index='Model', columns=target_col_name, values=value_col_name)

    convert_df.to_csv(save_path, mode="w")
    
    print(convert_df)




if __name__ == "__main__":
    ###  base setting for model testing ### 
    # convert_to_visual_assets ('./results_v1_new_robot_oob-mobilenet_v3_large-fold_1-last/Patient_Total_metric-ROBOT-results_v1_new_robot_oob-mobilenet_v3_large-fold_1-last.csv', 'mobilenet_v3_large', 'Confidence_Ratio', './results_v1_new_robot_oob-mobilenet_v3_large-fold_1-last/PIVOT/Patinet_CR-ROBOT-results_v1_new_robot_oob-mobilenet_v3_large-fold_1-last.csv')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    test_start()
    
