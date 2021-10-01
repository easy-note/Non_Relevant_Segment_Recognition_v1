### only using shared evaluation module as same as VIHUB PRO
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

from core.api.Inference_vihub.VIHUB_pro_QA_v2.test import inference_by_frame_list # inference module
from core.api.Evaluation_vihub.OOB_inference_module import Inference_eval # evaluation module
from core.api.Evaluation_vihub.OOB_Post_Procssing_module import FilterBank # PP module
from core.utils.ffmpegHelper import ffmpegHelper # ffmpeg module
from core.utils.evalHelper import evalHelper # eval Helper
from core.utils.visualHelper import visualHelper # visual Helper

parser = argparse.ArgumentParser()

parser.add_argument('--save_path', type=str, help='saveing results path for json')
parser.add_argument('--frame_dir', type=str, help='frame_dir')
parser.add_argument('--results_path', type=str, help='saved results path for predict vector')
parser.add_argument('--gt_path', type=str, help='annotation json path')

args, _ = parser.parse_known_args()

'''
# EVALUATION RESULT FOLDER HIREACHY
    results/
        mobilenet/
            mobilenet.json # <- gt, predict length diff -> ???? # input predict list, gt list 
        efficientnet/
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
def evaluation_vihub_results(results_path, gt_path, frame_dir, save_path):

    # 1.set path
    ## set save results dir
    mobilenet_results_dir = os.path.join(save_path, 'mobilenet')
    efficientnet_results_dir = os.path.join(save_path, 'efficient')
    os.makedirs(mobilenet_results_dir, exist_ok=True)
    os.makedirs(efficientnet_results_dir, exist_ok=True)

    visual_dir = os.path.join(save_path, 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    INFERENCE_STEP = 30

    ## set results file path
    mobilenet_predict_path = os.path.join(results_path, 'mobile.csv')
    efficientnet_predict_path = os.path.join(results_path, 'efficient.csv')

    mobilenet_eval_path = os.path.join(mobilenet_results_dir, 'mobilenet.json')
    efficientnet_eval_path = os.path.join(efficientnet_results_dir, 'efficientnet.json')


    ##### evaluation Helper [strat] #####
    mobile_evalHelper = evalHelper(mobilenet_predict_path, gt_path, inference_step=INFERENCE_STEP) # mobilenet
    efficient_evalHelper = evalHelper(efficientnet_predict_path, gt_path, inference_step=INFERENCE_STEP) # efficient

    gt_list, mobilenet_predict_list= mobile_evalHelper.load_gt_list_predict_list()
    gt_list, efficient_predict_list= efficient_evalHelper.load_gt_list_predict_list()
    ##### evaluation Helper [end] #####

    ## [2] evaluation module
    print('mobile : ', Inference_eval(mobilenet_predict_path, gt_path, mobilenet_eval_path, inference_step=INFERENCE_STEP).calc_OR_CR()) # mobilenet
    print('efficient : ', Inference_eval(efficientnet_predict_path, gt_path, efficientnet_eval_path, inference_step=INFERENCE_STEP).calc_OR_CR()) # efficient

    ## [3] visual module
    visual_helper = visualHelper(gt_list, mobilenet_predict_list, efficient_predict_list, visual_dir, frame_path=frame_dir) # visual (mobile, efficient)
    visual_helper.inference_visual()
    visual_helper.generate_gif() # FP FN gif (mobile, efficient)


if __name__ == '__main__':
    evaluation_vihub_results(args.results_path, args.gt_path, args.frame_dir, args.save_path)



