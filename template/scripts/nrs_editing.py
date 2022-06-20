
def extract_target_video_path(target_video_path):
    import os
    '''
    target_video_path = '/data3/Public/ViHUB-pro/input_video/train_100/R_2/01_G_01_R_2_ch1_01.mp4'
    core_output_path = 'train_100/R_2/01_G_01_R_2_ch1_01'
    '''

    split_path_list = target_video_path.split('/')
    edited_path_list = []

    for i, split_path in enumerate(split_path_list):
        if split_path == 'input_video':
            edited_path_list = split_path_list[i+1:]

    core_output_path = '/'.join(edited_path_list)
    core_output_path = core_output_path.split('.')[0]

    return core_output_path


def get_video_meta_info(target_video, base_output_path):
    import os
    import glob
    import datetime

    inference_json_output_path = os.path.join(base_output_path, 'inference_json', extract_target_video_path(target_video))

    video_name = target_video.split('/')[-1]
    video_path = target_video    
    frame_list = glob.glob(os.path.join(inference_json_output_path, 'frames', '*.jpg'))
    date_time = str(datetime.datetime.now())

    return video_name, video_path, len(frame_list), date_time


def save_meta_log(target_video, base_output_path):
    import json
    from collections import OrderedDict

    import os
    '''
    	"04_GS4_99_L_1_01.mp4": {
            "video_path": "/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01.mp4",
            "frame_cnt": 108461,
            "date": "2022-01-06 17:55:36.291518"
	}
    '''

    print('\nmeta log saving ...')

    meta_log_path = os.path.join(base_output_path, 'logs')
    os.makedirs(meta_log_path, exist_ok=True)
    
    meta_data = OrderedDict()
    video_name, video_path, frame_cnt, date_time = get_video_meta_info(target_video, base_output_path)

    meta_data[video_name] = {
        'video_path': video_path,
        'frame_cnt': frame_cnt,
        'date': date_time
    }

    print(json.dumps(meta_data, indent='\t'))

    try: # existing editing_log.json 
        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log.json'), 'r+') as f:
            data = json.load(f)
            data.update(meta_data)

            f.seek(0)
            json.dump(data, f, indent=2)
    except:
        os.makedirs(os.path.join(meta_log_path, 'editing_logs'), exist_ok=True)

        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log.json'), 'w') as f:
            json.dump(meta_data, f, indent=2)



# 22.01.07 hg modify, 기존 target_video 내부에 생성 => 설정된 frame_save_path 생성되도록 변경
def frame_cutting(target_video, frame_save_path):
    import sys
    import os    

    print('\nframe cutting ... ')

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')

    from core.utils.ffmpegHelper import ffmpegHelper

    # save_path 
    os.makedirs(frame_save_path, exist_ok=True)

    # frame cutting -> save to '$ frame_save_path~/frame-000000000.jpg'
    ffmpeg_helper = ffmpegHelper(target_video, frame_save_path)
    ffmpeg_helper.cut_frame_total()

    return frame_save_path


def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    # args.save_path = args.save_path + '-model:{}-IB_ratio:{}-WS_ratio:{}-hem_extract_mode:{}-top_ratio:{}-seed:{}'.format(args.model, args.IB_ratio, args.WS_ratio, args.hem_extract_mode, args.top_ratio, args.random_seed) # offline method별 top_ratio별 IB_ratio별 실험을 위해
    args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args


# 22.01.17 hg modify, InfernceDB독립사용을 위해 target_video 변수를 target_dir로 변경하고 infernce_interval parameter를 추가하였습니다.
def inference(target_dir, inference_interval, result_save_path, model_path, video_name):
    '''
    inference
    result_save_path : /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01/results/04_GS4_99_L_1_01.json
    추후 정량 평가 가능하면, results 내에 결과 저장. (CR, OR 등)
    '''
    import os
    import pandas as pd
    import glob
    import sys

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')
    sys.path.append(base_path+'/core/accessory/RepVGG')

    from core.utils.ffmpegHelper import ffmpegHelper

    ### inference_main
    ### test inference module
    from core.api.trainer import CAMIO
    from core.api.inference import InferenceDB # inference module
    # 22.01.17 hg modify, 해당 inference 함수 사용시 사용하지 않는 module import문을 제거하였습니다.

    print('\ninferencing ...')

    # 22.01.17 hg comment, args는 model 불러올떄만 사용하고, InferencDB사용시에 args 사용을 제거하였습니다.
    args = get_experiment_args()

    model = CAMIO.load_from_checkpoint(model_path, args=args) # ckpt
    model = model.cuda()

    # Inference module
    inference = InferenceDB(args, model, target_dir, inference_interval) # 22.01.17 hg modify, InferenceDB 사용시 기존 args.inference_interval 에서 function param으로 변경하였습니다.
    predict_list, target_img_list, target_frame_idx_list, ib_list, oob_list = inference.start() # call start

    print('\nlen(predict_list)', len(predict_list))
    print('len(target_img_list)', len(target_img_list))
    print('len(target_frame_idx_list)', len(target_frame_idx_list))
    print('len(ib_list) + len(oob_list)', len(ib_list) + len(oob_list), '\n')

    # save predict list to csv
    # video_name = result_save_path.split('/')[-1]

    '''
    # except (img_db length 와 json length 다를 때 처리)
    if len(target_frame_idx_list) > len(ib_list)+len(oob_list):
        target_frame_idx_list = target_frame_idx_list[:len(ib_list)+len(oob_list)]

    if len(target_img_list) > len(ib_list)+len(oob_list):
        target_img_list = target_img_list[:len(ib_list)+len(oob_list)]

    '''
    
    predict_csv_path = os.path.join(result_save_path, '{}.csv'.format(video_name))
    predict_df = pd.DataFrame({
                    'frame_idx': target_frame_idx_list,
                    'predict': predict_list,
                    'target_img': target_img_list,
                    'ib_list_softmax' : ib_list,
                    'oob_list_softmax' : oob_list
                })

    predict_df.to_csv(predict_csv_path)

    return predict_csv_path


def get_event_sequence_from_csv(predict_csv_path):
    import pandas as pd

    return pd.read_csv(predict_csv_path)['predict'].values.tolist()


def get_video_meta_info_from_ffmpeg(video_path):
    from core.utils.ffmpegHelper import ffmpegHelper

    print('\n\n \t\t <<<<< GET META INFO FROM FFMPEG >>>>> \t\t \n\n')

    ffmpeg_helper = ffmpegHelper(video_path)
    fps = ffmpeg_helper.get_video_fps()
    video_len = ffmpeg_helper.get_video_length()
    width, height = ffmpeg_helper.get_video_resolution()

    return fps, video_len, width, height


def extract_video_clip(video_path, results_dir, start_time, duration):
    from core.utils.ffmpegHelper import ffmpegHelper

    ffmpeg_helper = ffmpegHelper(video_path, results_dir)
    ffmpeg_helper.extract_video_clip(start_time, duration)


def parse_clips_paths(clips_root_dir):
    import os
    import glob
    from natsort import natsort

    target_clip_paths = glob.glob(os.path.join(clips_root_dir, 'clip-*.mp4'))
    target_clip_paths = natsort.natsorted(target_clip_paths)

    save_path = os.path.join(clips_root_dir, 'clips.txt')

    logging = ''

    for clip_path in target_clip_paths:
        txt = 'file \'{}\''.format(os.path.abspath(clip_path))
        logging += txt + '\n'
    
    print(logging)

    # save txt
    with open(save_path, 'w') as f :
        f.write(logging)

    return save_path


def clips_to_video(clips_root_dir, merge_path):
    from core.utils.ffmpegHelper import ffmpegHelper

    # parsing clip video list 
    input_txt_path = parse_clips_paths(clips_root_dir)

    ffmpeg_helper = ffmpegHelper("dummy", "dummy")
    ffmpeg_helper.merge_video_clip(input_txt_path, merge_path)


## 4. 22.01.07 hg new add, save annotation by inference (hvat form)
def report_annotation(frameRate, totalFrame, width, height, name, event_sequence, inference_interval, result_save_path):
    from core.utils.report import ReportAnnotation
    from core.utils.misc import get_nrs_frame_chunk, get_current_time
    
    ### static meta data ###
    createdAt = get_current_time()[0]
    updatedAt = createdAt
    _id = "temp"
    annotationType = "NRS"
    annotator = "temp"
    label = {"1": "NonRelevantSurgery"}
    ### ### ### ### ### ###

    nrs_frame_chunk = get_nrs_frame_chunk(event_sequence, inference_interval)

    annotation_report = ReportAnnotation(result_save_path) # load Report module

    # set meta info
    annotation_report.set_total_report(totalFrame, frameRate, width, height, _id, annotationType, createdAt, updatedAt, annotator, name, label)
    
    # add nrs annotation info
    nrs_cnt = len(nrs_frame_chunk)
    for i, (start_frame, end_frame) in enumerate(nrs_frame_chunk, 1):
        # print('\n\n[{}] \t {} - {}'.format(i, start_frame, end_frame))

        # check over totalFrame on last annotation (because of quntization? when set up inference_interval > 1)
        if nrs_cnt == i and end_frame >= totalFrame: 
            end_frame = totalFrame - 1

        annotation_report.add_annotation_report(start_frame, end_frame, code=1)

    annotation_report.save_report()

## 5. 22.01.07 hg write, inference_interval, video_fps는 clips의 extration time 계산시 필요
def video_editing(video_path, event_sequence, editted_video_path, inference_interval, video_fps):
    import os
    from core.utils.misc import get_rs_time_chunk

    print('\nvideo editing ...')

    # temp process path
    temp_process_dir = os.path.dirname(editted_video_path) # clips, 편한곳에 생성~(현재는 edit되는 비디오 밑에 ./clips에서 작업)
    temp_clip_dir = os.path.join(temp_process_dir, 'clips')
    os.makedirs(temp_clip_dir, exist_ok=True)

    # 0. inference vector(sequence) to rs chunk
    target_clipping_time = get_rs_time_chunk(event_sequence, video_fps, inference_interval)


    print('\n\n \t\t <<<<< EXTRACTING CLIPS >>>>> \t\t \n\n')
    # 1. clip video from rs chunk
    for i, (start_time, duration) in enumerate(target_clipping_time, 1):
        print('\n\n[{}] \t {} - {}'.format(i, start_time, duration))
        extract_video_clip(video_path, temp_clip_dir, start_time, duration)

    # 2. merge video
    print('\n\n \t\t <<<<< MERGEING CLIPS >>>>> \t\t \n\n')
    clips_to_video(clips_root_dir = temp_clip_dir, merge_path = editted_video_path)

    # 3. TODO: delete temp process path (delete clips)


def video_copy_to_save_dir(target_video, output_path):
    import os
    import shutil

    print('\nVideo copying ...')

    video_name = output_path.split('/')[-1]
    video_ext = target_video.split('.')[-1]

    video_name = video_name + '.' + video_ext

    # copy target_video
    print('COPY {} \n==========> {}\n'.format(target_video, os.path.join(output_path, video_name)))
    shutil.copy2(target_video, os.path.join(output_path, video_name))
    

def check_exist_dupli_video(target_video, inference_json_output_path, edited_video_output_path):
    import os

    if os.path.exists(inference_json_output_path) and os.path.exists(edited_video_output_path):
        return True
    
    return False

# predict csv to applied-pp predict csv
def apply_post_processing(predict_csv_path, seq_fps):
    import os
    import pandas as pd
    from core.api.post_process import FilterBank
    
    # 1. load predict df
    predict_df = pd.read_csv(predict_csv_path, index_col=0)
    event_sequence = predict_df['predict'].tolist()

    # 2. apply pp sequence
    #### use case 1. best filter
    fb = FilterBank(event_sequence, seq_fps)
    best_pp_seq_list = fb.apply_best_filter()

    #### use case 2. custimize filter
    '''
    best_pp_seq_list = fb.apply_filter(event_sequence, "opening", kernel_size=3)
    best_pp_seq_list = fb.apply_filter(best_pp_seq_list, "closing", kernel_size=3)
    '''

    predict_df['predict'] = best_pp_seq_list

    # 3. save pp df
    d_, f_ = os.path.split(predict_csv_path)
    f_name, _ = os.path.splitext(f_)

    results_path = os.path.join(d_, '{}-pp.csv'.format(f_name)) # renaming file of pp
    predict_df.to_csv(results_path)

    return results_path



def main(model_path, save_dir):
    import os
    import glob
    import re

    import pandas as pd

    import natsort
    import json

    from core.utils.metric import MetricHelper
    from core.api.evaluation import Evaluator
       
    ## --- robot ---
    # fold1 = ['R_2', 'R_6', 'R_13', 'R_74', 'R_100', 'R_202', 'R_301', 'R_302', 'R_311', 'R_312', 
    #       'R_313', 'R_336', 'R_362', 'R_363', 'R_386', 'R_405', 'R_418', 'R_423', 'R_424', 'R_526']
    # img_path = '/raid/NRS/robot/mola/img'
    # json_path = '/raid/NRS/robot/mola/anno/v3'


    ## --- robot_etc ---
    # fold1 = ['R_6', 'R_46', 'R_154', 'R_155', 'R_156', 'R_157', 'R_158','R_160', 'R_161', 'R_162', 'R_163', 'R_164', 'R_165','R_166' , 'R_167', 'R_168']
    # img_path = '/raid/OOB_Recog/img_db/ETC24'
    # json_path = '/raid/NRS/robot/etc/anno/v3'


    ## --- lapa ---
    fold1 = ['01_VIHUB1.2_A9_L_5', '01_VIHUB1.2_A9_L_6', '01_VIHUB1.2_A9_L_18', '01_VIHUB1.2_A9_L_19', '01_VIHUB1.2_A9_L_20', \
            '01_VIHUB1.2_A9_L_21', '01_VIHUB1.2_A9_L_24', '01_VIHUB1.2_A9_L_27', '01_VIHUB1.2_A9_L_30', '01_VIHUB1.2_A9_L_35', \
            '01_VIHUB1.2_A9_L_38', '01_VIHUB1.2_A9_L_40', '01_VIHUB1.2_A9_L_47', '01_VIHUB1.2_A9_L_48', '01_VIHUB1.2_A9_L_51', \
            '01_VIHUB1.2_B4_L_1', '01_VIHUB1.2_B4_L_2', '01_VIHUB1.2_B4_L_4', '01_VIHUB1.2_B4_L_6', '01_VIHUB1.2_B4_L_7', \
            '01_VIHUB1.2_B4_L_8', '01_VIHUB1.2_B4_L_9', '01_VIHUB1.2_B4_L_12', '01_VIHUB1.2_B4_L_16', '01_VIHUB1.2_B4_L_17', \
            '01_VIHUB1.2_B4_L_20', '01_VIHUB1.2_B4_L_22', '01_VIHUB1.2_B4_L_24', '01_VIHUB1.2_B4_L_26', '01_VIHUB1.2_B4_L_28', \
            '01_VIHUB1.2_B4_L_29', '01_VIHUB1.2_B4_L_75', '01_VIHUB1.2_B4_L_76', '01_VIHUB1.2_B4_L_82', '01_VIHUB1.2_B4_L_84', \
            '01_VIHUB1.2_B4_L_86', '01_VIHUB1.2_B4_L_87', '01_VIHUB1.2_B4_L_90', '01_VIHUB1.2_B4_L_91', '01_VIHUB1.2_B4_L_94', \
            '01_VIHUB1.2_B4_L_98', '01_VIHUB1.2_B4_L_100', '01_VIHUB1.2_B4_L_103', '01_VIHUB1.2_B4_L_106', '01_VIHUB1.2_B4_L_107', \
            '01_VIHUB1.2_B4_L_108', '01_VIHUB1.2_B4_L_111', '01_VIHUB1.2_B4_L_113', '01_VIHUB1.2_B4_L_115', '01_VIHUB1.2_B4_L_120', \
            '01_VIHUB1.2_B4_L_121', '01_VIHUB1.2_B4_L_123', '01_VIHUB1.2_B4_L_127', '01_VIHUB1.2_B4_L_130', '01_VIHUB1.2_B4_L_131', \
            '01_VIHUB1.2_B4_L_134', '01_VIHUB1.2_B4_L_136', '01_VIHUB1.2_B4_L_137', '01_VIHUB1.2_B4_L_139', '01_VIHUB1.2_B4_L_143', \
            '01_VIHUB1.2_B4_L_144', '01_VIHUB1.2_B4_L_146', '01_VIHUB1.2_B4_L_149', '01_VIHUB1.2_B4_L_150', '01_VIHUB1.2_B4_L_151', \
            '01_VIHUB1.2_B4_L_152', '01_VIHUB1.2_B4_L_153', '01_VIHUB1.2_B5_L_9', '04_GS4_99_L_3', '04_GS4_99_L_4', '04_GS4_99_L_7', \
            '04_GS4_99_L_11', '04_GS4_99_L_12', '04_GS4_99_L_16', '04_GS4_99_L_17', '04_GS4_99_L_26', '04_GS4_99_L_28', '04_GS4_99_L_29', \
            '04_GS4_99_L_37', '04_GS4_99_L_38', '04_GS4_99_L_39', '04_GS4_99_L_40', '04_GS4_99_L_42', '04_GS4_99_L_44', '04_GS4_99_L_46', \
            '04_GS4_99_L_48', '04_GS4_99_L_49', '04_GS4_99_L_50', '04_GS4_99_L_58', '04_GS4_99_L_59', '04_GS4_99_L_60', '04_GS4_99_L_61', \
            '04_GS4_99_L_64', '04_GS4_99_L_65', '04_GS4_99_L_66', '04_GS4_99_L_71', '04_GS4_99_L_72', '04_GS4_99_L_75', '04_GS4_99_L_79', \
            '04_GS4_99_L_82', '04_GS4_99_L_84', '04_GS4_99_L_86', '04_GS4_99_L_87', '04_GS4_99_L_88', '04_GS4_99_L_89', '04_GS4_99_L_92', \
            '04_GS4_99_L_94', '04_GS4_99_L_95', '04_GS4_99_L_96', '04_GS4_99_L_98', '04_GS4_99_L_99', '04_GS4_99_L_102', '04_GS4_99_L_103', \
            '04_GS4_99_L_104', '04_GS4_99_L_106', '04_GS4_99_L_107', '04_GS4_99_L_108', '04_GS4_99_L_114', '04_GS4_99_L_116', '04_GS4_99_L_117',\
            '04_GS4_99_L_120', '04_GS4_99_L_125', '04_GS4_99_L_126']

    img_path = '/raid/NRS/lapa/vihub/img'
    json_path = '/raid/NRS/lapa/vihub/anno/v3'


    # 추후 args로 받아올 경우 해당 변수를 args. 로 초기화
    seq_fps = 1 # pp module (1 fps 로 inference) - pp에서 사용 (variable이 고정되어 30 fps인 비디오만 사용하는 시나이오로 적용, 비디오에 따라 유동적으로 var이 변하도록 계산하는게 필요해보임)
    inference_interval = 30 # frame inference interval - 전체사용 (variable이 고정되어 30 fps인 비디오를 1fps로 Infeence 하는 시나리오로 적용, 비디오에 따라 유동적으로 var이 변하도록 계산하는게 필요해보임)

    model_path = model_path
    
    jsons = glob.glob(os.path.join(json_path, '*.json'))

    # anno: 01_GS3_06_R_171_01_TBE_31.json
    # frame: raid/OOB_Recog/img_db/ETC24/R_154/R_154_empty_01/01_GS3_06_R_154_empty_01-0000052726.jpg

    for patient_no in fold1:
        target_path = os.path.join(img_path, patient_no)

        for (root, dirs, files) in os.walk(target_path):
            dirs = natsort.natsorted(dirs)

            # dirs = dirs[:int(len(dirs)/2)]

            for dir in dirs:
                #frames = glob.glob(os.path.join(root, dir, '*.jpg'))
                #frames = natsort.natsorted(frames)

                # 비디오 전처리 (frmae 추출) -> 임시 디렉토리
                frame_save_path = os.path.join(root, dir)
                
                t_patient = frame_save_path.split('/')[-2]
                t_video_no = frame_save_path.split('_')[-1]

                # t_patient = frame_save_path.split('/')[-2]
                # t_video_no = frame_save_path.split('/')[-1].split('_')[-1]

                print('frame_save_path ==>', frame_save_path, '\n')

                for i in jsons:
                    ## --- robot (etc) ---
                    # patient = '_'.join(i.split('_')[3:5])
                    # video_no = i.split('_')[5]
                    
                    ## --- lapa ---
                    patient = '_'.join(os.path.basename(i).split('_')[:5])
                    video_no = '_'.join(i.split('_')[5:6])

                    if patient == t_patient and video_no == t_video_no:
                        target_json = i

                
                # # inference (비디오 단위) -> 저장 디렉토리 & result csv 생성 
                # result_save_path = os.path.join('./220510-rebuttal-3', t_patient)
                result_save_path = os.path.join(save_dir, t_patient)
                # video_name = target_json.split('/')[-1].split('.')[0]
                video_name = os.path.splitext(os.path.basename(target_json))[0]
                
                os.makedirs(result_save_path, exist_ok=True)
                predict_csv_path = inference(target_dir = frame_save_path, inference_interval = inference_interval, result_save_path = result_save_path, model_path = model_path, video_name=video_name) 

                er = Evaluator(predict_csv_path, target_json, inference_interval)
                gt_list, _ = er.get_assets()

                final_df = pd.read_csv(predict_csv_path)

                predict_len = len(final_df['predict'])

                if len(gt_list) > predict_len:
                    print('len(gt_list) > predict_len')
                    gt_list = gt_list[:predict_len]

                else:
                    final_df = pd.DataFrame({
                        'frame_idx': final_df['frame_idx'].tolist()[:len(gt_list)],
                        'predict': final_df['predict'].tolist()[:len(gt_list)],
                        'target_img': final_df['target_img'].tolist()[:len(gt_list)],
                        'ib_list_softmax' : final_df['ib_list_softmax'].tolist()[:len(gt_list)],
                        'oob_list_softmax' : final_df['oob_list_softmax'].tolist()[:len(gt_list)]
                    })

                    print('final_df')
                    print(final_df)

                final_df['gt'] = gt_list
                final_df.to_csv(os.path.join(result_save_path, video_name+'-gt.csv'))
                
                
                # # # metrics = er.calc()

# plot no skill and model precision-recall curves
def plot_pr_curve(test_y, model_probs,clf_name):
    # calculate the no skill line as the proportion of the positive class
    print(test_y[test_y==1])
    exit(0)
    no_skill = len(test_y[test_y==1]) / len(test_y)
    # plot the no skill precision-recall curve
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(testy, model_probs)
    pyplot.plot(recall, precision, marker='.', label=clf_name)
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    pyplot.savefig('./test.png')

def cal_metric():
    import os
    import glob

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc

    import scikitplot as skplt
    import matplotlib.pyplot as plt

    import pandas as pd
    import natsort

    # base_path = '/OOB_RECOG/template/scripts/220511-HEM_off'
    # base_path = '/OOB_RECOG/template/scripts/220511-RS'
    # base_path = '/OOB_RECOG/template/scripts/220511-WS'
    base_path = '/OOB_RECOG/template/scripts/220511-theator'

    patient_score = []


    for (root, dirs, files) in os.walk(base_path):
        dirs = natsort.natsorted(dirs)
        
        for dir in dirs:        
            patient_path = os.path.join(root, dir)
            patient_no = patient_path.split('/')[-1]
            csv_list = glob.glob(os.path.join(patient_path, '*-gt.csv'))

            df_list = []
            for csv in csv_list:    
                df = pd.read_csv(csv)
                df_list.append(df)

            paient_total = pd.concat(df_list)

            gt = paient_total['gt'].tolist()

            model_predict = paient_total['oob_list_softmax'].tolist()
            predict_list = paient_total['predict']
            oob_predict = paient_total['oob_list_softmax'].tolist()
            ib_predict = paient_total['ib_list_softmax'].tolist()

            model_predicts = []
            for i, o in zip(ib_predict, oob_predict):
                model_predicts.append([i,o])

            # roc_auc_score_i = roc_auc_score(gt, model_predict)
            roc_auc_score_i = roc_auc_score(gt, model_predict)

            print(patient_no, ': roc_auc_score', roc_auc_score_i)

            # plot roc curves
            skplt.metrics.plot_roc_curve(gt, model_predicts)
            os.makedirs('results-{}'.format(base_path.split('/')[-1]), exist_ok=True)
            plt.savefig('results-{}/{}-roc_curve.png'.format(base_path.split('/')[-1], patient_no))

            patient_score.append([patient_no, roc_auc_score_i])

    with open('./results-{}/roc_score.csv'.format(base_path.split('/')[-1]), 'w',newline='') as f: 

        import csv
      
        # using csv.writer method from CSV package 
        write = csv.writer(f) 
        write.writerows(patient_score)






if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core')

    import os
    
    
    ### --- robot etc ----
    # # RS
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/rs-epoch=41-Mean_metric=0.9911-best.ckpt'
    # save_dir = './220511-etc-RS'

    # # WS
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/ws-epoch=17-Mean_metric=0.9923-best.ckpt'
    # save_dir = './220511-etc-WS'

    # # Theator
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/theator-epoch=45-Mean_metric=0.9913-best.ckpt'
    # save_dir = './220511-etc-theator'

    # offline
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/hem-off-epoch=5-Mean_metric=0.9895-best.ckpt'
    # save_dir = './220511-etc-offline'

    # Online
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/hem-on-epoch=78-Mean_metric=0.9814-best.ckpt'
    # save_dir = './220511-etc-online'


    ### --- lapa ----
    # RS (4)
    os.environ["CUDA_VISIBLE_DEVICES"]= "7"
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/logs-lapa/rs-epoch=74-Mean_metric=0.9879-best.ckpt'
    model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/logs-lapa/rs_epoch=99-val_loss=0.0175-last.ckpt'
    save_dir = './lapa_results/lapa-RS-last'

    # # WS (5)
    # os.environ["CUDA_VISIBLE_DEVICES"]= "5"
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/logs-lapa/ws-epoch=32-Mean_metric=0.9861-best.ckpt'
    # save_dir = './lapa_results/lapa-WS'

    # # offline (6)
    # os.environ["CUDA_VISIBLE_DEVICES"]= "7"
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/logs-lapa/hem-off-stage1-epoch=48-Mean_metric=0.9892-best.ckpt'
    # save_dir = './lapa_results/0617-lapa-offline'

    # ## Online (7)
    # model_path = '/OOB_RECOG/template/2022_miccai_rebuttal/logs-lapa/hem-on-epoch=95-Mean_metric=0.9868-best.ckpt'
    # save_dir = './lapa_results/lapa-online'

    main(model_path, save_dir)

    