### only using shared inference module as same as VIHUB PRO
if __package__ is None:
    import sys
    import os

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_path)
    sys.path.append(os.path.join(root_path, 'core'))
    sys.path.append(os.path.join(root_path, 'core', 'api'))
    sys.path.append(os.path.join(root_path, 'core', 'api', 'Evaluation_vihub'))
    sys.path.append(os.path.join(root_path, 'core', 'api', 'Inference_vihub'))
    sys.path.append(os.path.join(root_path, 'core', 'api', 'Inference_vihub', 'VIHUB_pro_QA_v2'))

    sys.path.append(os.path.join(root_path, 'core', 'utils'))

    print(sys.path)


import os
import sys
import argparse
import pandas as pd
import json

from core.api.Inference_vihub.VIHUB_pro_QA_v2.test import inference_by_frame_list # inference module
from core.api.Evaluation_vihub.OOB_inference_module import Inference_eval # evaluation module
from core.api.Evaluation_vihub.OOB_Post_Procssing_module import FilterBank # PP module
from core.utils.ffmpegHelper import ffmpegHelper # ffmpeg module
from core.utils.evalHelper import evalHelper # eval Helper
from core.utils.visualHelper import visualHelper # visual Helper

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, help='model_root_path')
parser.add_argument('--video_path', type=str, help='input video path')
parser.add_argument('--results_path', type=str, help='saving results path for predict vector')
parser.add_argument('--gt_path', type=str, help='annotation json path')
parser.add_argument('--pp', action='store_true', help='If True, apply pp')

args, _ = parser.parse_known_args()


'''
# INFERENCE RESULT FOLDER HIREACHY
    results/
        frames/
            frame00
            frame01 ...
            ...
        mobilenet/
            mobilenet.csv -> p1 path
            mobilenet.json # <- gt, predict length diff -> ???? # input predict list, gt list 
        efficientnet/
            efficientnet.csv -> p2 path
            efficientnet.json # <- gt, predict length diff -> ???? # input predict list, gt list
        visual/
            visual.png
            mobilenet/
                mobilenet_FP.gif
                mobilenet_FN.gif
            efficientnet/
                efficientnet_FP.gif
                efficientnet_FN.gif
'''
def inference_using_module(model_path, video_path, results_path, gt_path, pp):

    # 1.set path
    ## set model path
    mobilenet_model_path = os.path.join(model_path, 'mobilenet_v3_large.ckpt')
    efficientnet_model_path = os.path.join(model_path, 'efficientnet_b3.ckpt')

    ## set cutted frame path

    cutted_frame_path = os.path.join(results_path, 'frame')
    os.makedirs(cutted_frame_path, exist_ok=True)

    ## set results dir
    mobilenet_results_dir = os.path.join(results_path, 'mobilenet')
    efficientnet_results_dir = os.path.join(results_path, 'efficient')
    os.makedirs(mobilenet_results_dir, exist_ok=True)
    os.makedirs(efficientnet_results_dir, exist_ok=True)

    visual_dir = os.path.join(results_path, 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    ## set results file path
    mobilenet_predict_path = os.path.join(mobilenet_results_dir, 'mobilenet.csv')
    efficientnet_predict_path = os.path.join(efficientnet_results_dir, 'efficientnet.csv')

    mobilenet_eval_path = os.path.join(mobilenet_results_dir, 'mobilenet.json')
    efficientnet_eval_path = os.path.join(efficientnet_results_dir, 'efficientnet.json')

    # 2. frame cutting module
    ffmpeg = ffmpegHelper(video_path, cutted_frame_path)
    ffmpeg.cut_frame_1fps()

    # 3.module test
    ## [1] inference module
    mobilenet_predict_list = inference_by_frame_list(cutted_frame_path, mobilenet_model_path) # mobilenet
    efficient_predict_list = inference_by_frame_list(cutted_frame_path, efficientnet_model_path) # efficient

    ## (save predict ector to csv)
    pd.DataFrame({'predict': mobilenet_predict_list}).to_csv(mobilenet_predict_path)
    pd.DataFrame({'predict': efficient_predict_list}).to_csv(efficientnet_predict_path)

    print('----')
    mobilenet_predict_list = list(pd.read_csv(mobilenet_predict_path)['predict'])
    efficient_predict_list = list(pd.read_csv(efficientnet_predict_path)['predict'])

    ##### evaluation Helper [strat] #####
    mobile_evalHelper = evalHelper(mobilenet_predict_path, gt_path, inference_step=30) # mobilenet
    efficient_evalHelper = evalHelper(efficientnet_predict_path, gt_path, inference_step=30) # efficient


    gt_list, mobilenet_predict_list= mobile_evalHelper.load_gt_list_predict_list()
    gt_list, efficient_predict_list= efficient_evalHelper.load_gt_list_predict_list()
    ##### evaluation Helper [end] #####

    print(mobilenet_predict_list)

    ## [1.5] pp module
    if pp == True :
        print('== APPLY PP ==')
        mobilenet_predict_list = FilterBank(mobilenet_predict_list, seq_fps=1).apply_best_filter()
        efficient_predict_list = FilterBank(efficient_predict_list, seq_fps=1).apply_best_filter()
        print('== PP DONE ==')

    ## [2] evaluation module
    mobilenet_metric_json = Inference_eval(mobilenet_predict_path, gt_path, inference_step=30).calc_OR_CR() # mobilenet
    efficientnet_metric_json = Inference_eval(efficientnet_predict_path, gt_path, inference_step=30).calc_OR_CR() # efficient

    ## (save metric to json)
    with open(mobilenet_eval_path, "w") as json_file:
        json.dump(mobilenet_metric_json, json_file)
    
    with open(efficientnet_eval_path, "w") as json_file:
        json.dump(efficientnet_metric_json, json_file)

    ## [3] visual module
    visual_helper = visualHelper(gt_list, mobilenet_predict_list, efficient_predict_list, visual_dir, frame_path=cutted_frame_path) # visual (mobile, efficient)
    visual_helper.inference_visual()
    visual_helper.generate_gif() # FP FN gif (mobile, efficient)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    inference_using_module(args.model_path, args.video_path, args.results_path, args.gt_path, args.pp)


