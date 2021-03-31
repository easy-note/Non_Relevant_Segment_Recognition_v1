import os
import cv2
from PIL import Image
import torch
import numpy as np
import pandas as pd
import glob
import matplotlib
import argparse

from pandas import DataFrame as df
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch

from Model import CAMIO
from torchvision import transforms

import datetime

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score





def test_video() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str,
                        default=os.getcwd() + '/logs/robot/OOB/robot_oob_train_3/epoch=2-val_loss=0.0541.ckpt', help='trained model_path')
    # robot_3 | 
    # robot_4 | /logs/robot/OOB/robot_oob_train_4/epoch=9-val_loss=0.0597.ckpt
    

    # os.getcwd() + '/logs/robot/OOB/robot_oob_train_1/epoch=12-val_loss=0.0303.ckpt' // OOB = 1
    # os.getcwd() + '/logs/OOB_robot_test7/epoch=6-val_loss=0.0323.ckpt' // OOB = 0
    parser.add_argument('--data_dir', type=str,
                        default='/data/CAM_IO/robot/video', help='video_path :) ')

    parser.add_argument('--anno_dir', type=str,
                        default='/data/CAM_IO/robot/OOB', help='annotation_path :) ')

    parser.add_argument('--results_save_dir', type=str,
                        default=os.path.join(os.getcwd(), 'results3'), help='inference results save path')

    parser.add_argument('--mode', type=str,
                        default='robot', choices=['robot', 'lapa'], help='inference results save path')

    args, _ = parser.parse_known_args()

    print(args)


    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # dirscirbe exception, inference setting for each mode
    if args.mode == 'lapa' :        
        # test_video_for_lapa()
        exit(0) # not yet support
        pass
    
    else : # robot
        model = CAMIO()
        model = model.load_from_checkpoint(args.model_path)

        model.cuda()
        model.eval()

        

        
        

        print('\n\t=== model_loded for ROBOT ===\n')

        # starting inference
        test_video_for_robot(args.data_dir, args.anno_dir, args.results_save_dir, model, data_transforms)





### union def ###
# cal vedio frame
def time_to_idx(time, fps):
    t_segment = time.split(':')
    # idx = int(t_segment[0]) * 3600 * fps + int(t_segment[1]) * 60 * fps + int(t_segment[2])
    idx = (int(t_segment[0]) * 3600 * fps) + (int(t_segment[1]) * 60 * fps) + (int(t_segment[2]) * fps) # [h, m, s, ms] 

    return idx

def idx_to_time(idx, fps) :
    time_s = idx // fps
    converted_time = str(datetime.timedelta(seconds=time_s))

    return converted_time

# check results for binary metric
def calc_confusion_metric(gts, preds):

    classification_report_result = classification_report(gts, preds, target_names=['IB', 'OOB'], zero_division=0)
    
    prec = precision_score(gts, preds, average='binary',pos_label=1, zero_division=0)
    recall = recall_score(gts, preds, average='binary',pos_label=1, zero_division=0)

    metric = pd.crosstab(pd.Series(gts), pd.Series(preds), rownames=['True'], colnames=['Predicted'], margins=True)
    
    saved_text = '{} \nprecision \t : \t {} \nrecall \t\t : \t {} \n\n{}'.format(classification_report_result, prec, recall, metric)

    return saved_text

### union def ### 


### for robot def ###
def test_video_for_robot(data_dir, anno_dir, results_save_dir, model, data_transforms) :
    
    ### base setting ###
    # valset = ['R017', 'R022', 'R116', 'R208', 'R303']
    valset = ['R017']

    video_ext = '.mp4'
    fps = 30

    tar_surgery = 'robot'

    ### ### ###

    # gettering information step
    info_dict = gettering_information_for_robot(data_dir, anno_dir, valset, fps, video_ext)

    print('\n\n\t ==== RESULTS OF GETTERING INFORMATION==== ')
    print('\tSUCESS GETTERING VIDEO SET: ', len(info_dict['video']))
    print('\tSUCESS GETTERING ANNOTATION SET: ', len(info_dict['anno']))
    print('\t=== === === ===\n\n')

    # inference step
    inference_for_robot(info_dict, model, data_transforms, results_save_dir, inference_step=5)
    
    

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

def inference_for_robot(info_dict, model, data_transforms, results_save_dir, inference_step=1) : 
    print('\n\n\n\t\t\t ### STARTING DEF [inference_for_robot] ### \n\n')

    # create results folder
    try :
        if not os.path.exists(results_save_dir) :
            os.makedirs(results_save_dir)
    except OSError :
        print('ERROR : Creating Directory, ' + results_save_dir)

    total_videoset_cnt = len(info_dict['video']) # total number of video set

    # loop from total_videoset_cnt
    for i, (video_path_list, anno_info_list) in enumerate(zip(info_dict['video'], info_dict['anno']), 1):
        videoset_name = os.path.basename(video_path_list[0]).split('_')[0] # parsing videoset name

        # create base folder for save results each video set
        each_videoset_result_dir = os.path.join(results_save_dir, videoset_name) # '~~~/results/R022' , '~~~/results/R011' ..
        try :
            if not os.path.exists(os.path.join(each_videoset_result_dir)) :
                os.makedirs(each_videoset_result_dir)
        except OSError :
            print('ERROR : Creating Directory, ' + each_videoset_result_dir)


        print('COUNT OF VIDEO SET | {} / {} \t\t ======>  VIDEO SET | {}'.format(i, total_videoset_cnt, videoset_name))
        print('NUMBER OF VIDEO : {} | NUMBER OF ANNOTATION INFO : {}'.format(len(video_path_list), len(anno_info_list)))
        print('RESULTS SAVED AT \t\t\t ======>  {}'.format(each_videoset_result_dir))
        print('\n')
        
        for video_path, anno_info in zip(video_path_list, anno_info_list) :
            
            video_name = os.path.basename(video_path).split('.')[0] # only video name

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

            # open video cap
            video = cv2.VideoCapture(video_path)
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            print('\tTarget video : {} | Total Frame : {}'.format(video_name, video_len))
            print('\tAnnotation Info : {}'.format(anno_info))

            ### check idx -> time
            for start, end in anno_info :
                print([idx_to_time(start, 30), idx_to_time(end, 30)])

            ###
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

            # init_variable
            gt_list = [] # ground truth
            predict_list = [] # predict
            frame_idx_list = [] # frmae info
            time_list = [] # frame to time

            FP_frame_cnt = 0
            FN_frame_cnt = 0
            frame_check_cnt = 0 # loop cnt

            # inference per frame 
            for frame_idx, truth in enumerate(tqdm(truth_list, desc='Inferencing... \t ==> {}'.format(video_name))) :
                predict = -1
                img = None
                _img = None

                if frame_idx % inference_step != 0 :
                    continue

                video.set(1, frame_idx) # frame setting
                _, img = video.read() # read frame

                # inference frame in model
                _img = data_transforms['val'](Image.fromarray(img))
                _img = torch.unsqueeze(_img, 0).cuda()
                outputs = model(_img)

                # results of predict
                predict = torch.argmax(outputs.cpu()) # predict

                # save results
                frame_idx_list.append(frame_idx)
                gt_list.append(truth)
                predict_list.append(int(predict))
                time_list.append(idx_to_time(frame_idx, 30))
                # predict_list.append(predict.data.numpy())
                
                # saving FP
                if truth == IB_CLASS and predict == OOB_CLASS :
                    FP_frame_cnt+=1
                    print('frame no {} | truth {} | predict {}'.format(frame_idx, truth, predict))
                    print('CHECKED_FRAME_CNT({}) | [FP:FN] = [{}:{}]'.format(frame_check_cnt, FP_frame_cnt, FN_frame_cnt))

                    fp_frame_saving_path = os.path.join(fp_frame_saved_dir, '{}_{:010d}.jpg'.format(video_name, frame_idx))
                    print('Saving FP Frame \t\t ', fp_frame_saving_path)
                    cv2.imwrite(fp_frame_saving_path, img)
                    print('')

                # saving FN
                if truth == OOB_CLASS and predict == IB_CLASS :
                    FN_frame_cnt+=1
                    print('frame no {} | truth {} | predict {}'.format(frame_idx, truth, predict))
                    print('CHECKED_FRAME_CNT({}) | [FP:FN] = [{}:{}]'.format(frame_check_cnt, FP_frame_cnt, FN_frame_cnt))

                    fn_frame_saving_path = os.path.join(fn_frame_saved_dir, '{}_{:010d}.jpg'.format(video_name, frame_idx))
                    print('Saving FN Frame \t\t ', fn_frame_saving_path)
                    cv2.imwrite(fn_frame_saving_path, img)
                    print('')
                
                frame_check_cnt+=1 # loop cnt check

            print('TOTAL FRAME : ', len(frame_idx_list))
            print('FP FRAME CNT : ', FP_frame_cnt)
            print('FN FRAME CNT : ', FN_frame_cnt)
            
            video.release()

            # saving inferece result
            result_dict = {
                'frame' : frame_idx_list,
                'time' : time_list,
                'truth' : gt_list,
                'predict' : predict_list
            }

            inference_results_df = pd.DataFrame(result_dict)
            
            print('Result Saved at \t ====> ', each_video_result_dir)
            inference_results_df.to_csv(os.path.join(each_video_result_dir, 'Inference_{}.csv'.format(video_name)), mode="w")

            # saving plot
            fig = plt.figure(figsize=(16,8))

            # plt.hold()
            plt.scatter(np.array(frame_idx_list), np.array(gt_list), color='blue', marker='o', s=15, label='Truth') # ground truth
            plt.scatter(np.array(frame_idx_list), np.array(predict_list), color='red', marker='o', s=5, label='Predict') # predict

            plt.title('Inference Results By per {} Frame | Video : {}'.format(inference_step, video_name));
            plt.suptitle('OOB_CLASS [{}] | IB_CLASS [{}] | FP : {} | FN : {}'.format(OOB_CLASS, IB_CLASS, FP_frame_cnt, FN_frame_cnt));
            plt.ylabel('class'); plt.xlabel('frame');
            plt.legend(loc='center right');

            plt.savefig(os.path.join(each_video_result_dir, 'plot_{}.png'.format(video_name)));

            # save metric
            # calc_confusion_metric(inference_results_df['truth'], inference_results_df['predict'])
            saved_text = calc_confusion_metric(gt_list, predict_list)

            with open(os.path.join(each_video_result_dir, 'Metric_{}.txt'.format(video_name)), 'w') as f :
                f.write(saved_text)






if __name__ == "__main__":
    ###  base setting for model testing ### 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    test_video()