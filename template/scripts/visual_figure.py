def visual_for_prediction_results(patient_gt_list, patient_no, model_name, results_dict, save_dir):
     # figure b, predict results per methods
    from core.api.visualization import VisualTool # visual module

    import os
    import pandas as pd
    import glob

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'prediction_results_of_{}.png'.format(patient_no))

    visual_tool = VisualTool(patient_gt_list, patient_no, save_path)
    visual_tool.visual_predict_multi(results_dict, model_name, inference_interval=30, window_size=300, section_num=2)


def get_inference_results_per_patient(inference_results_dir, patient_no):
    import os

    from core.utils.parser import FileLoader
    from core.utils.prepare import PatientsGT

    file_loader = FileLoader()    

    patient_gt = PatientsGT()
    videos = patient_gt.get_video_no(patient_no)

    patient_predict_results_dir = os.path.join(inference_results_dir, patient_no)

    patient_predict_list = []

    for video_no in videos:
        video_predict_results_path = os.path.join(patient_predict_results_dir, video_no, '{}.csv'.format(video_no))
        
        file_loader.set_file_path(video_predict_results_path)
        predict_df = file_loader.load()

        patient_predict_list += predict_df['predict'].tolist()

    return patient_predict_list


def extract_frame_by_index(patient_no, frame_index, save_dir):
    ### extract one frame by frame_index // frame_index = 500
    from core.utils.ffmpegHelper import ffmpegHelper

    os.makedirs(save_dir, exist_ok=True)
    ffmpeg_helper = ffmpegHelper('/data1/HuToM/Video_Robot_cordname/R000021/ch1_video_01.mp4', save_dir)

    ffmpeg_helper.extract_frame_by_index(frame_index=frame_index)

def extract_frame_by_time(patient_no, time, save_dir):
    ### extract one frame by frame_index // frame_index = 500
    from core.utils.ffmpegHelper import ffmpegHelper

    os.makedirs(results_dir, exist_ok=True)
    ffmpeg_helper = ffmpegHelper('/data1/HuToM/Video_Robot_cordname/R000021/ch1_video_01.mp4', save_dir)

    ffmpeg_helper.extract_frame_by_time(time=time)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        import os
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core/accessory/RepVGG')
        print(base_path)
        
        from core.utils.misc import save_dict_to_csv, prepare_inference_aseets, get_inference_model_path, \
                set_args_per_stage, check_hem_online_mode, clean_paging_chache

        from core.utils.parser import FileLoader
        from core.utils.prepare import PatientsGT
    '''
    ### figure b ####
    patient_no = 'R_2'
    model_name = 'mobilenet_v3_large'
    save_dir = '/OOB_RECOG/figures/predictions'

    patient_gt = PatientsGT()
    patient_gt_list = patient_gt.get_gt(patient_no)
    patient_gt_list = patient_gt_list[::30]

    results_dict = {
        '(a)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs', 'mobilenet_apply_offline_methods-all-offline-IB_ratio=3-ws_ratio=3-MC=5-experiment-model:mobilenetv3_large_100-IB_ratio:3.0-WS_ratio:3-hem_extract_mode:all-offline-top_ratio:0.07-seed:3829', 'TB_log', 'version_4', 'inference_results'), patient_no),
        '(b)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs', 'mobilenet_apply_offline_methods-all-offline-IB_ratio=3-ws_ratio=3-MC=5-experiment-model:mobilenetv3_large_100-IB_ratio:3.0-WS_ratio:3-hem_extract_mode:all-offline-top_ratio:0.07-seed:3829', 'TB_log', 'version_5', 'inference_results'), patient_no),
        '(c)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs', 'mobilenet_apply_offline_methods-all-offline-IB_ratio=3-ws_ratio=3-MC=5-experiment-model:mobilenetv3_large_100-IB_ratio:3.0-WS_ratio:3-hem_extract_mode:all-offline-top_ratio:0.07-seed:3829', 'TB_log', 'version_6', 'inference_results'), patient_no),
        '(d)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs', 'mobilenet_apply_offline_methods-all-offline-IB_ratio=3-ws_ratio=3-MC=5-experiment-model:mobilenetv3_large_100-IB_ratio:3.0-WS_ratio:3-hem_extract_mode:all-offline-top_ratio:0.07-seed:3829', 'TB_log', 'version_7', 'inference_results'), patient_no),
        '(e)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs', 'mobilenet_apply_offline_methods-all-offline-IB_ratio=3-ws_ratio=3-MC=5-experiment-model:mobilenetv3_large_100-IB_ratio:3.0-WS_ratio:3-hem_extract_mode:all-offline-top_ratio:0.07-seed:3829', 'TB_log', 'version_7', 'inference_results'), patient_no),
        '(f)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs', 'mobilenet_apply_offline_methods-all-offline-IB_ratio=3-ws_ratio=3-MC=5-experiment-model:mobilenetv3_large_100-IB_ratio:3.0-WS_ratio:3-hem_extract_mode:all-offline-top_ratio:0.07-seed:3829', 'TB_log', 'version_7', 'inference_results'), patient_no),
    }

    visual_for_prediction_results(patient_gt_list, patient_no, model_name, results_dict, save_dir) # figure b, predict results per methods
    '''

    ### gradcam ####
    patinet_no = 'R_1'
    frame_index = 300
    time = '00:00:10.00'
    save_dir = '/OOB_RECOG/ffmpeg_results'

    patient_gt = PatientsGT()
    patient_gt.get_video_no(patient_no)

    patinet_
    patient_gt.get_start_idx()
    

    # extract_frame_by_index(patinet_no, frame_index, save_dir)
    extract_frame_by_time(patinet_no, time, save_dir)



    







