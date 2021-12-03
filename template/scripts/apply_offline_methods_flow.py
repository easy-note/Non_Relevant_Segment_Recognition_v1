# STAGE_LIST = ['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3', 'hem_train', 'general_train']
# STAGE_LIST=['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3', 'hem_train', 'hem_train', 'hem_train', 'hem_train', 'hem_train'] # general 완성전까지 hem_train까지만 진행
STAGE_LIST=['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3', 'hem_train', 'hem_train', 'hem_train', 'hem_train', 'hem_train'] # general 완성전까지 hem_train까지만 진행

def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    # args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    args.save_path = args.save_path + '-model:{}-IB_ratio:{}-WS_ratio:{}-hem_extract_mode:{}-top_ratio:{}-seed:{}'.format(args.model, args.IB_ratio, args.WS_ratio, args.hem_extract_mode, args.top_ratio, args.random_seed) # offline method별 top_ratio별 IB_ratio별 실험을 위해
    # args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args

def train_main(args):
    print('train_main')
    
    import os
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.plugins import DDPPlugin

    from core.model import get_model, get_loss
    from core.api.trainer import CAMIO
    from core.api.theator_trainer import TheatorTrainer

    from torchsummary import summary

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save_path,
        name='TB_log',
        default_hp_metric=False)

    if args.experiment_type == 'theator':
        x = TheatorTrainer(args)
    elif args.experiment_type == 'ours':
        x = CAMIO(args)
    
    if args.num_gpus > 1:
        trainer = pl.Trainer(gpus=args.num_gpus, 
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,
                            plugins=DDPPlugin(find_unused_parameters=False), # [Warning DDP] error ?
                            accelerator='ddp')
    else:
        if args.use_test_batch:
            trainer = pl.Trainer(gpus=args.num_gpus,
                            limit_train_batches=2,
                            limit_val_batches=2,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)
        else:    
            trainer = pl.Trainer(gpus=args.num_gpus,
                            # limit_train_batches=2,#0.01,
                            # limit_val_batches=2,#0.01,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)
    
    trainer.fit(x)

    # args.restore_path = os.path.join(args.save_path, 'TB_log', 'version_0') # TO DO: we should define restore path
    
    args.restore_path = os.path.join(x.restore_path)
    print('restore_path: ', args.restore_path)
    
    return args

def inference_main(args):
    print('inference_main')
    
    ### test inference module
    from core.api.trainer import CAMIO
    from core.api.theator_trainer import TheatorTrainer
    from core.api.inference import InferenceDB # inference module
    from core.api.evaluation import Evaluator # evaluation module
    from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)
    from core.utils.logger import Report # report helper (for experiments reuslts and inference results)

    from core.api.visualization import VisualTool # visual module
    
    import os
    import pandas as pd
    import glob

    # from pretrained model
    '''
    model = CAMIO(args)
    model = model.cuda()
    '''

    print('restore : ', args.restore_path)

    # from finetuning model
    if 'repvgg' not in args.model:
        model_path = get_inference_model_path(os.path.join(args.restore_path, 'checkpoints'))
        
        if args.experiment_type == 'theator':
            model = TheatorTrainer.load_from_checkpoint(model_path, args=args)
        elif args.experiment_type == 'ours':
            model = CAMIO.load_from_checkpoint(model_path, args=args)
    else:
        if args.experiment_type == 'theator':
            model = TheatorTrainer(args)
        elif args.experiment_type == 'ours':
            model = CAMIO(args)
        
    model = model.cuda()

    # inference block
    os.makedirs(args.restore_path, exist_ok=True)

    inference_assets_save_path = args.restore_path
    details_results_path = os.path.join(args.restore_path, 'inference_results')
    patients_results_path = os.path.join(details_results_path, 'patients_report.csv')
    videos_results_path = os.path.join(details_results_path, 'videos_report.csv')

    report_path = os.path.join(args.restore_path, 'Report.json')


    # 1. load inference dataset
    # inference case(ROBOT, LAPA), anno_ver(1,2,3), inference fold(1,2,3,4,5)에 따라 환자별 db. gt_json 잡을 수 있도록 set up
    case = args.dataset # ['ROBOT', LAPA]
    anno_ver = '3'
    inference_fold = args.inference_fold

    inference_assets = prepare_inference_aseets(case=case , anno_ver=anno_ver, inference_fold=inference_fold, save_path=inference_assets_save_path)
    patients = inference_assets['patients']
    
    patients_count = len(patients)

    # 2. for record metrics
    # Report
    report = Report(report_path)

    patients_metrics_list = [] # save each patients metrics
    for idx in range(patients_count): # per patients
        patient = patients[idx]
        patient_no = patient['patient_no']
        patient_video = patient['patient_video']

        videos_metrics_list = [] # save each videos metrics
        
        patient_gt_list = [] # each patients gt list for visual
        patient_predict_list = [] # each patients predict list for visual

        # for save patients results
        each_patients_save_dir = os.path.join(details_results_path, patient_no)
        os.makedirs(each_patients_save_dir, exist_ok=True)

        for video_path_info in patient['path_info']: # per videos
            video_name = video_path_info['video_name']
            video_path = video_path_info['video_path']
            annotation_path = video_path_info['annotation_path']
            db_path = video_path_info['db_path']

            # Inference module
            inference = InferenceDB(model, db_path, args.inference_interval) # Inference object
            predict_list, target_img_list, target_frame_idx_list = inference.start() # call start
  
            # for save video results
            each_videos_save_dir = os.path.join(each_patients_save_dir, video_name)
            os.makedirs(each_videos_save_dir, exist_ok=True)

            # save predict list to csv
            predict_csv_path = os.path.join(each_videos_save_dir, '{}.csv'.format(video_name))
            predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'target_img': target_img_list,
                        })
            predict_df.to_csv(predict_csv_path)

            # Evaluation module
            evaluator = Evaluator(predict_csv_path, annotation_path, args.inference_interval)
            gt_list, predict_list = evaluator.get_assets() # get gt_list, predict_list by inference_interval

            # save predict list to csv
            predict_csv_path = os.path.join(each_videos_save_dir, '{}.csv'.format(video_name))
            predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'gt': gt_list,
                            'target_img': target_img_list,
                        })
            predict_df.to_csv(predict_csv_path)

            # for visulization per patients
            patient_gt_list += gt_list
            patient_predict_list += predict_list

            # metric per video
            video_metrics = evaluator.calc() # same return with metricHelper
            video_CR, video_OR = video_metrics['CR'], video_metrics['OR']
            video_TP, video_FP, video_TN, video_FN = video_metrics['TP'], video_metrics['FP'], video_metrics['TN'], video_metrics['FN']
            video_TOTAL = video_FP + video_TP + video_FN + video_TN

            video_gt_IB, video_gt_OOB, video_predict_IB, video_predict_OOB = video_metrics['gt_IB'], video_metrics['gt_OOB'], video_metrics['predict_IB'], video_metrics['predict_OOB']

            video_jaccard = video_metrics['Jaccard']

            print('\t => video_name: {}'.format(video_name))
            print('\t    video_CR: {:.3f} | video_OR: {:.3f}'.format(video_CR, video_OR))
            print('\t    video_TP: {} | video_FP: {} | video_TN: {} | video_FN: {}'.format(video_TP, video_FP, video_TN, video_FN))

            # save video metrics
            video_results_dict = report.add_videos_report(patient_no=patient_no, video_no=video_name, FP=video_FP, TP=video_TP, FN=video_FN, TN=video_TN, TOTAL=video_TOTAL, CR=video_CR, OR=video_OR, gt_IB=video_gt_IB, gt_OOB=video_gt_OOB, predict_IB=video_predict_IB, predict_OOB=video_predict_OOB, jaccard=video_jaccard)
            save_dict_to_csv(video_results_dict, videos_results_path)

            # for calc patients metric
            videos_metrics_list.append(video_metrics)
        
        # calc each patients CR, OR
        patient_metrics = MetricHelper().aggregate_calc_metric(videos_metrics_list)
        patient_CR, patient_OR = patient_metrics['CR'], patient_metrics['OR']
        patient_TP, patient_FP, patient_TN, patient_FN = patient_metrics['TP'], patient_metrics['FP'], patient_metrics['TN'], patient_metrics['FN']
        patient_TOTAL = patient_FP + patient_TP + patient_FN + patient_TN

        patient_gt_IB, patient_gt_OOB, patient_predict_IB, patient_predict_OOB = patient_metrics['gt_IB'], patient_metrics['gt_OOB'], patient_metrics['predict_IB'], patient_metrics['predict_OOB']

        patient_jaccard = patient_metrics['Jaccard']

        print('\t\t => patient_no: {}'.format(patient_no))
        print('\t\t    patient_CR: {:.3f} | patient_OR: {:.3f}'.format(patient_CR, patient_OR))
        print('\t\t    patient_TP: {} | patient_FP: {} | patient_TN: {} | patient_FN: {}'.format(patient_TP, patient_FP, patient_TN, patient_FN))

        # save patient metrics        
        patient_results_dict = report.add_patients_report(patient_no=patient_no, FP=patient_FP, TP=patient_TP, FN=patient_FN, TN=patient_TN, TOTAL=patient_TOTAL, CR=patient_CR, OR=patient_OR, gt_IB=patient_gt_IB, gt_OOB=patient_gt_OOB, predict_IB=patient_predict_IB, predict_OOB=patient_predict_OOB, jaccard=patient_jaccard)
        save_dict_to_csv(patient_results_dict, patients_results_path)
    
        # for calc total patients CR, OR
        patients_metrics_list.append(patient_metrics)
        
        # visualization per patients
        patient_predict_visual_path = os.path.join(each_patients_save_dir, 'predict-{}.png'.format(patient_no))

        visual_tool = VisualTool(patient_gt_list, patient_no, patient_predict_visual_path)
        visual_tool.visual_predict(patient_predict_list, args.model, args.inference_interval, window_size=300, section_num=2)

        # CLEAR PAGING CACHE
        # clean_paging_chache()

    # for calc total patients CR, OR + (mCR, mOR)
    total_metrics = MetricHelper().aggregate_calc_metric(patients_metrics_list)
    total_mCR, total_mOR, total_CR, total_OR = total_metrics['mCR'], total_metrics['mOR'], total_metrics['CR'], total_metrics['OR']

    report.set_experiment(model=args.model, methods=args.hem_extract_mode, inference_fold=args.inference_fold, mCR=total_mCR, mOR=total_mOR, CR=total_CR, OR=total_OR, details_path=details_results_path, model_path=model_path)
    report.save_report() # save report

    # SUMMARY
    patients_CR = report.get_patients_CR()
    patients_OR = report.get_patients_CR()

    experiment_summary = {
        'model':args.model,
        'methods':args.hem_extract_mode,
        'top_ratio':args.top_ratio,
        'stage': args.stage,

        'random_seed': args.random_seed,
        'IB_ratio': args.IB_ratio,
        'WS_ratio': args.WS_ratio,
        'inference_fold':args.inference_fold,

        'mCR':total_mCR,
        'mOR':total_mOR,
        'CR':total_CR,
        'OR':total_OR,
        'details_path':details_results_path,
        'model_path': model_path,
    }
    
    # return mCR, mOR, OR, CR of experiment
    return args, experiment_summary, patients_CR, patients_OR

def apply_offline_methods_main(args):
    import glob
    import os
    import json

    from core.api.trainer import CAMIO
    from core.dataset.robot_dataset import RobotDataset
    from core.config.assets_info import mc_assets_save_path
    from core.dataset.hem_methods import HEMHelper

    logger_path = os.path.join(args.save_path, 'TB_log') # tb_logger path

    restore_path = {
        'mini_fold_stage_0':os.path.join(logger_path, 'version_0'),
        'mini_fold_stage_1':os.path.join(logger_path, 'version_1'),
        'mini_fold_stage_2':os.path.join(logger_path, 'version_2'),
        'mini_fold_stage_3':os.path.join(logger_path, 'version_3'),
    }

    args.restore_path = restore_path[args.stage] # set restore_path
    os.makedirs(args.restore_path, exist_ok=True) # for saveing version 0,1,2,3


    # /data2/Public/OOB_Recog/offline/models/mobilenetv3_large_100/WS=2-IB=3-seed=3829/mini_fold_stage_0/last/n_dropout=5
    # /data2/Public/OOB_Recog/offline/models/mobilenetv3_large_100/WS=2-IB=3-seed=3829/mini_fold_stage_0
    model_dir = os.path.join(mc_assets_save_path['robot'], args.model, 'WS={}-IB={}-seed={}'.format(args.WS_ratio, int(args.IB_ratio), args.random_seed), args.stage)

    # 1-1. model 불러오기
    if 'repvgg' not in args.model:
        model_path = get_inference_model_path(model_dir) # best, last 결정
        
        if args.experiment_type == 'theator':
            model = TheatorTrainer.load_from_checkpoint(model_path, args=args)
        elif args.experiment_type == 'ours':
            model = CAMIO.load_from_checkpoint(model_path, args=args)
    else:
        if args.experiment_type == 'theator':
            model = TheatorTrainer(args)
        elif args.experiment_type == 'ours':
            model = CAMIO(args)
        
    model = model.cuda()
    # model.eval() # 어차피 mc dropout 에서 처리

    # 1-2. train/validation set 불러오기 // train set 불러오는 이유는 hem extract 할때 얼마나 뽑을지 정해주는 DATASET_COUNT.json을 저장하기 위해
    trainset = RobotDataset(args, state='train') # train dataset setting
    
    args.use_all_sample = True
    valset = RobotDataset(args, state='val') # val dataset setting
    args.use_all_sample = False

    transet_rs_count, trainset_nrs_count = trainset.number_of_rs_nrs()
    valset_rs_count, valset_nrs_count = valset.number_of_rs_nrs()
            
    train_save_data = {
        'train_dataset': {
            'rs': transet_rs_count,
            'nrs': trainset_nrs_count 
        },
        'target_hem_count': {
            'rs': transet_rs_count // 3,
            'nrs': trainset_nrs_count // 3
        }
    }

    val_save_data = {
        'val_dataset': {
            'rs': valset_rs_count,
            'nrs': valset_nrs_count 
        },
    }

    patient_per_dic = valset.number_of_patient_rs_nrs()

    val_save_data.update(patient_per_dic)

    with open(os.path.join(args.restore_path, 'DATASET_COUNT.json'), 'w') as f:
        json.dump(train_save_data, f, indent=2)

    with open(os.path.join(args.restore_path, 'PATIENTS_DATASET_COUNT.json'), 'w') as f:
        json.dump(val_save_data, f, indent=2)

    # 2. hem_methods 적용
    hem_helper = HEMHelper(args)
    hem_helper.set_method(args.hem_extract_mode) # 'all-offline'
    hem_helper.set_restore_path(args.restore_path)
    # softmax_diff_small_hem_df, softmax_diff_large_hem_df, voting_hem_df, vi_small_hem_df, vi_large_hem_df = hem_helper.compute_hem(model, valset)
    softmax_diff_large_hem_df = hem_helper.compute_hem(model, valset)

    # 3. hem_df.to_csv
    # softmax_diff_small_hem_df.to_csv(os.path.join(args.restore_path, 'softmax_diff_small_{}-{}-{}.csv'.format(args.model, args.hem_extract_mode, args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
    softmax_diff_large_hem_df.to_csv(os.path.join(args.restore_path, 'softmax_diff_large_{}-{}-{}.csv'.format(args.model, args.hem_extract_mode, args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)

    # voting_hem_df.to_csv(os.path.join(args.restore_path, 'voting_{}-{}-{}.csv'.format(args.model, args.hem_extract_mode, args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
            
    # vi_small_hem_df.to_csv(os.path.join(args.restore_path, 'vi_small_{}-{}-{}.csv'.format(args.model, args.hem_extract_mode, args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
    # vi_large_hem_df.to_csv(os.path.join(args.restore_path, 'vi_large_{}-{}-{}.csv'.format(args.model, args.hem_extract_mode, args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)

    '''
    if self.args.hem_extract_mode == 'all-offline':
    version_dict = {
        '3': ['softmax_diff_small_*-*-*.csv', 'softmax_diff_small_hem_assets.csv'],
        '4': ['softmax_diff_large_*-*-*.csv', 'softmax_diff_large_hem_assets.csv'],
        '5': ['voting_*-*-*.csv', 'voting_hem_assets.csv'],
        '6': ['vi_small_*-*-*.csv', 'vi_small_hem_assets.csv'],
        '7': ['vi_large_*-*-*.csv', 'vi_large_hem_assets.csv'],
    }
    num = self.args.restore_path.split('_')[-1]
            
    load_f_path, save_f_path = version_dict[num]
    '''

    return args


def main():    
    # 0. set each experiment args 
    import os, torch, random
    import numpy as np

    # 0. set each experiment args 
    args = get_experiment_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"]=str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True


    # general mode
    if args.stage == 'general_train':
        args.mini_fold = 'general'
        args = train_main(args)
        
        # 3. inference
        args, experiment_summary, patients_CR, patients_OR = inference_main(args)

        # 4. save experiments summary
        experiments_sheet_path = os.path.join(args.experiments_sheet_dir, 'experiments_summary-fold_{}.csv'.format(args.inference_fold))
        os.makedirs(args.experiments_sheet_dir, exist_ok=True)

        save_dict_to_csv({**experiment_summary, **patients_CR}, experiments_sheet_path)
    else: # online mode
        if check_hem_online_mode(args):
            args.mini_fold = 'general'
            args.stage = 'hem_train'

            args = train_main(args)

            # 3. inference
            args, experiment_summary, patients_CR, patients_OR = inference_main(args)

            # 4. save experiments summary
            experiments_sheet_path = os.path.join(args.experiments_sheet_dir, 'experiments_summary-fold_{}.csv'.format(args.inference_fold))
            os.makedirs(args.experiments_sheet_dir, exist_ok=True)

            save_dict_to_csv({**experiment_summary, **patients_CR}, experiments_sheet_path)
        # offline mode
        else:
            for ids, stage in enumerate(STAGE_LIST):

                if idx == 5:
                    exit(0)

                args = set_args_per_stage(args, ids, stage) # 첫번째 mini-fold 1 

                print('\n\n')
                print('====='*7, args.stage.upper(),'====='*7)
                print('\n\n')

                print(args)

                if ids > 3:
                    args = train_main(args)
                else:
                    args = apply_offline_methods_main(args)
                    
                if ids > 3:
                    # 3. inference
                    if ids == 4: # version 4
                        # args.hem_extract_mode = 'hem-softmax-small_diff-offline'
                        args.hem_extract_mode = 'hem-softmax-large_diff-offline'
                    elif ids == 5: # version 5
                        args.hem_extract_mode = 'hem-softmax-large_diff-offline'

                    elif ids == 6: # version 6
                        args.hem_extract_mode = 'hem-voting-offline'

                    elif ids == 7: # version 7
                        args.hem_extract_mode = 'hem-vi_small-offline'
                    elif ids == 8: # version 8
                        args.hem_extract_mode = 'hem-vi_large-offline'
                
                    args, experiment_summary, patients_CR, patients_OR = inference_main(args)

                    # 4. save experiments summary
                    experiments_sheet_path = os.path.join(args.experiments_sheet_dir, 'experiments_summary-fold_{}.csv'.format(args.inference_fold))
                    os.makedirs(args.experiments_sheet_dir, exist_ok=True)

                    save_dict_to_csv({**experiment_summary, **patients_CR}, experiments_sheet_path)
                    
                    args.hem_extract_mode = 'all-offline'

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core/accessory/RepVGG')
        print(base_path)
        
        from core.utils.misc import save_dict_to_csv, prepare_inference_aseets, get_inference_model_path, \
    set_args_per_stage, check_hem_online_mode, clean_paging_chache

    main()

