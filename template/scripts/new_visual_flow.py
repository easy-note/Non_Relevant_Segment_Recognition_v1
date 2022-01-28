# STAGE_LIST = ['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3', 'hem_train', 'general_train']
# STAGE_LIST = ['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3'] #, 'hem_train', 'hem_train', 'hem_train', 'hem_train', 'hem_train'] # general 완성전까지 hem_train까지만 진행

def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    # args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    # args.save_path = args.save_path + '-model:{}-IB_ratio:{}-WS_ratio:{}-hem_extract_mode:{}-top_ratio:{}-seed:{}'.format(args.model, args.IB_ratio, args.WS_ratio, args.hem_extract_mode, args.top_ratio, args.random_seed) # offline method별 top_ratio별 IB_ratio별 실험을 위해

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

    from core.api.trainer import CAMIO
    from torchsummary import summary

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save_path,
        name='TB_log',
        default_hp_metric=False)

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
    model_path = get_inference_model_path(os.path.join(args.restore_path, 'checkpoints'))
    model = CAMIO.load_from_checkpoint(model_path, args=args) # .ckpt

    '''
    pt_path=None # for using change_deploy_mode for offline, it will be update on above if's branch

    if 'repvgg' in args.model: # load pt from version/checkpoints 
        pt_path = get_pt_path(os.path.join(args.restore_path, 'checkpoints'))
        print('\n\t ===> LOAD PT FROM {}\n'.format(pt_path))
    
    model.change_deploy_mode(pt_path=pt_path) # 이거는 repvgg나 multi일떄만 적용됨. offline시 repvgg는 저장된 Pt에서 불러와야 하므로 pt_path를 arguments로 넣어주어야 함. 
    '''
        
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
            inference = InferenceDB(args, model, db_path, args.inference_interval) # Inference object // args => InferenceDB init의 DBDataset(args) 생성시 args.model로 'mobile_vit' augmentation 처리해주기 위해
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

            video_precision, video_recall = video_metrics['Precision'], video_metrics['Recall']

            print('\t => video_name: {}'.format(video_name))
            print('\t    video_CR: {:.3f} | video_OR: {:.3f}'.format(video_CR, video_OR))
            print('\t    video_TP: {} | video_FP: {} | video_TN: {} | video_FN: {}'.format(video_TP, video_FP, video_TN, video_FN))

            # save video metrics
            video_results_dict = report.add_videos_report(patient_no=patient_no, video_no=video_name, FP=video_FP, TP=video_TP, FN=video_FN, TN=video_TN, TOTAL=video_TOTAL, CR=video_CR, OR=video_OR, gt_IB=video_gt_IB, gt_OOB=video_gt_OOB, predict_IB=video_predict_IB, predict_OOB=video_predict_OOB, precision=video_precision, recall=video_recall, jaccard=video_jaccard)
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

        patient_precision, patient_recall = patient_metrics['Precision'], patient_metrics['Recall']

        print('\t\t => patient_no: {}'.format(patient_no))
        print('\t\t    patient_CR: {:.3f} | patient_OR: {:.3f}'.format(patient_CR, patient_OR))
        print('\t\t    patient_TP: {} | patient_FP: {} | patient_TN: {} | patient_FN: {}'.format(patient_TP, patient_FP, patient_TN, patient_FN))

        # save patient metrics        
        patient_results_dict = report.add_patients_report(patient_no=patient_no, FP=patient_FP, TP=patient_TP, FN=patient_FN, TN=patient_TN, TOTAL=patient_TOTAL, CR=patient_CR, OR=patient_OR, gt_IB=patient_gt_IB, gt_OOB=patient_gt_OOB, predict_IB=patient_predict_IB, predict_OOB=patient_predict_OOB, precision=patient_precision, recall=patient_recall, jaccard=patient_jaccard)
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
    total_mPrecision, total_mRecall = total_metrics['mPrecision'], total_metrics['mRecall']
    total_Jaccard = total_metrics['Jaccard']

    report.set_experiment(model=args.model, methods=args.hem_extract_mode, inference_fold=args.inference_fold, mCR=total_mCR, mOR=total_mOR, CR=total_CR, OR=total_OR, mPrecision=total_mPrecision, mRecall=total_mRecall, Jaccard=total_Jaccard, details_path=details_results_path, model_path=model_path)
    report.save_report() # save report

    # SUMMARY
    patients_CR = report.get_patients_CR()
    patients_OR = report.get_patients_CR()

    experiment_summary = {
        'model':args.model,
        'methods':args.hem_extract_mode,
        'top_ratio':args.top_ratio,
        'stage': args.train_stage,

        'random_seed': args.random_seed,
        'IB_ratio': args.IB_ratio,
        'WS_ratio': args.WS_ratio,
        'inference_fold':args.inference_fold,

        'mCR':total_mCR,
        'mOR':total_mOR,
        'CR':total_CR,
        'OR':total_OR,
        'mPrecision':total_mPrecision,
        'mRecall': total_mRecall,
        'Jaccard': total_Jaccard,

        'details_path':details_results_path,
        'model_path': model_path,
    }
    
    # return mCR, mOR, OR, CR of experiment
    return args, experiment_summary, patients_CR, patients_OR

def get_hem_assets_path(hem_interation_idx, model_name, hem_extract_mode):
    import os
    from core.config.assets_info import mc_assets_save_path

    # model 과 hem_extract_mode, hem_extract_mode 만 변경
    # hem_interation_idx = 200 - 100
    # model_name = 'mobilenetv3_large_100'
    # hem_extract_mode='hem-softmax_diff_small-offline'
    
    # 무조건 고정항목
    WS_ratio = 3
    IB_ratio = 3
    n_dropout = 5
    random_seed = 3829

    # 각 model 별 hem extract mode 별 sota top_ratio  (ws, ib = 3, n_dropout = 5, random_seed = 3829로 고정했을 때) => 해당 setting 으로 뽑은 hem assets csv는 nas에 존재.
    top_ratio_sota_setup = {
        100:{
            'mobilenetv3_large_100':
                {
                    'hem-softmax_diff_small-offline': 0.05
                },
            
            'repvgg-a0':
                {
                    'hem-softmax_diff_small-offline': 0.05
                },

            'resnet18':
                {
                    'hem-softmax_diff_small-offline': 0.10
                },
        },

        200:{
            'mobilenetv3_large_100':
                {
                    'hem-softmax_diff_small-offline': 0.05
                },
            
            'repvgg-a0':
                {
                    'hem-softmax_diff_small-offline': 0.05 # 0.07도 있긴 함.
                },

            'resnet18':
                {
                    'hem-softmax_diff_small-offline': 0.10
                },
        },
    }

    top_ratio = top_ratio_sota_setup[hem_interation_idx-100][model_name][hem_extract_mode] # 없으면 error

    ## data2/../hem_assets/theator_stage_flag=100/resnet18/WS=3-IB=3-seed=3829/hem_extract_mode=hem-softmax_diff_small-offline-ver=1-top_ratio=5-n_dropout=5.csv
    hem_assets_path = os.path.join(mc_assets_save_path['robot'], 'hem_assets', 'theator_stage_flag={}'.format(hem_interation_idx - 100), model_name, 'WS={}-IB={}-seed={}'.format(int(WS_ratio), int(IB_ratio), random_seed), 'hem_extract_mode={}-ver=1-top_ratio={}-n_dropout={}.csv'.format(hem_extract_mode, int(top_ratio * 100) , n_dropout)) # you should get (theator_stage_flag - 100 dir)
    
    return hem_assets_path

def new_main():    
    # 0. set each experiment args 
    import os, torch, random
    import numpy as np

    from core.config.assets_info import mc_assets_save_path

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

    if args.multi_stage:
        print('Go Multi Stage!')
    else:
        args.n_stage = 1
        
    for N in range(args.n_stage):
        args.cur_stage = N+1

        # general mode
        if args.train_stage == 'general_train' or args.train_stage == 'hem_train':
            args = train_main(args)
            
            # 3. inference
            args, experiment_summary, patients_CR, patients_OR = inference_main(args)

            # 4. save experiments summary
            experiments_sheet_path = os.path.join(args.experiments_sheet_dir, 'experiments_summary-fold_{}.csv'.format(args.inference_fold))
            os.makedirs(args.experiments_sheet_dir, exist_ok=True)

            save_dict_to_csv({**experiment_summary, **patients_CR}, experiments_sheet_path)
    
        # baby model train (offline)
        else: # mini_fold_stage_0, 1, 2, 3
                mini_fold_stage = ['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3']

                if args.hem_interation_idx == 100:
                    args.appointment_assets_path = ''

                else: # 200, 300 일때는 appointmnet assets path 에서 csv 불러오기
                    # csv 불러오기 위한 용도로 scripts에서 받아오는 args.hem_iteration idx, model_name, hem_extract_mode 대한 의미를 다음과 같이 부여가능 => 이중의미 (실제로 baby model학습모델 + 이전 iteration 에서 model을 사용해서 뽑은 hem_assets csv))
                    hem_assets_path = get_hem_assets_path(args.hem_interation_idx, args.model, hem_extract_mode='hem-softmax_diff_small-offline') # hem_extract_mode 도 args로 받게 해도 됨.
                    args.appointment_assets_path = hem_assets_path
                
                for train_stage in mini_fold_stage:
                    args.train_stage = train_stage
                    args = train_main(args) # for iter 돌면서 args.restore_path가 이전 정보로 trainer init 될 것, 하지만 sanity check 이후 다시 restore_path 재설정하므로 해당 for문 괜찮을듯.


                
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        
        print('base path : ', base_path)
        
        from core.utils.misc import prepare_inference_aseets, get_inference_model_path, \
            clean_paging_chache, save_dict_to_csv, save_dataset_info

    new_main()

