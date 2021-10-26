def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    ### inference args
    parser.add_argument('--inference_save_dir', type=str, 
                        default='../results',
                        help='root directory for infernce saving')
        

    parser.add_argument('--inference_interval', type=int, 
                        default=300,
                        help='Inference Interval of frame')

    parser.add_argument('--inference_fold',
                    default='free',
                    type=str,
                    choices=['1', '2', '3', '4', '5', 'free'],
                    help='valset 1, 2, 3, 4, 5, free')

    parser.add_argument('--sampling_type',
                    default=1,
                    type=int,
                    choices=[1,2,3],)


    args = parser.parse_args()

    ### model basic info opts
    args.model = 'mobilenet_v3_large'
    args.pretrained = True
    args.sampling_type = 2

    ### dataset opts
    args.data_base_path = '/raid/img_db'
    args.train_method = 'normal' # ['normal', 'hem-softmax', 'hem-bs', 'hem-vi']
    args.batch_size = 128

    ### train args
    args.save_path = '/OOB_RECOG/logs/inference_project'
    args.num_gpus = 1
    args.max_epoch = 1
    args.min_epoch = 0

    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args

def get_inference_model_path(restore_path):
    # from finetuning model
    import glob

    ckpoint_path = os.path.join(restore_path, 'checkpoints', '*.ckpt')
    ckpts = glob.glob(ckpoint_path)
    
    for f_name in ckpts :
        if f_name.find('last') != -1 :
            return f_name



def train_main(args):
    print('train_main')
    
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

    x = CAMIO(args)
    print(summary(x.model, (3,224,224))) # check model arch
    # x = TheatorTrainer(args)
    '''

    if args.num_gpus > 1:
        trainer = pl.Trainer(gpus=args.num_gpus, 
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,
                            plugins=DDPPlugin(find_unused_parameters=False), # [Warning DDP] error ?
                            accelerator='ddp')
    else:
        trainer = pl.Trainer(gpus=args.num_gpus,
                            limit_train_batches=0.01,
                            limit_val_batches=0.01,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)

    trainer.fit(x)
    '''

    args.restore_path = os.path.join(args.save_path, 'TB_log', 'version_8') # TO DO: we should define restore path
    
    # args.restore_path = os.path.join(x.restore_path)
    print('restore_path: ', args.restore_path)
    
    return args

def clean_paging_chache():
    import subprocess # for CLEAR PAGING CACHE

    # clear Paging Cache because of I/O CACHE [docker run -it --name cam_io_hyeongyu -v /proc:/writable_proc -v /home/hyeongyuc/code/OOB_Recog:/OOB_RECOG -v /nas/OOB_Project:/data -p 6006:6006  --gpus all --ipc=host oob:1.0]
    print('\n\n\t ====> CLEAN PAGINGCACHE, DENTRIES, INODES "echo 1 > /writable_proc/sys/vm/drop_caches"\n\n')
    subprocess.run('sync', shell=True)
    subprocess.run('echo 1 > /writable_proc/sys/vm/drop_caches', shell=True) ### For use this Command you should make writable proc file when you run docker


def prepare_inference_aseets(case, anno_ver, inference_fold, save_path):
    from core.utils.prepare import InferenceAssets # inference assets helper (for prepare inference assets)
    from core.utils.prepare import OOBAssets # OOB assets helper (for prepare inference assets)
    from core.utils.parser import FileLoader # file load helper
    
    # OOBAssets
    assets_sheet_dir = os.path.join(save_path, 'assets')
    oob_assets = OOBAssets(assets_sheet_dir)
    # oob_assets.save_assets_sheet() # you can save assets sheet
    video_sheet, annotation_sheet, img_db_sheet = oob_assets.get_assets_sheet() # you can only use assets although not saving
    # video_sheet, annotation_sheet, img_db_sheet = oob_assets.load_assets_sheet(assets_sheet_dir) # you can also load saved aseets

    # InferenceAssets
    inference_assets_save_path = os.path.join(save_path, 'patients_aseets.yaml')
    inference_assets_helper = InferenceAssets(case=case, anno_ver=anno_ver, fold=inference_fold)
    inference_assets = inference_assets_helper.get_inference_assets() # dict (yaml)

    # save InferenceAssets: serialization from python object(dict) to YAML stream and save
    inference_assets_helper.save_dict_to_yaml(inference_assets, inference_assets_save_path)
    
    # load InferenceAssets: load saved inference assets yaml file // you can also load saved patients
    f_loader = FileLoader()
    f_loader.set_file_path(inference_assets_save_path)
    inference_assets = f_loader.load()

    return inference_assets


def inference_main(args):
    print('inference_main')
    
    ### test inference module
    from core.api.trainer import CAMIO
    from core.api.inference import InferenceDB # inference module
    from core.api.evaluation import Evaluator # evaluation module
    from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)
    from core.utils.logging import ReportHelper # report helper (for experiments reuslts and inference results)

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
    model_path = get_inference_model_path(args.restore_path)
    model = CAMIO.load_from_checkpoint(model_path, args=args)
    model = model.cuda()

    # inference block
    os.makedirs(args.inference_save_dir, exist_ok=True)

    # 1. load inference dataset
    # inference case(ROBOT, LAPA), anno_ver(1,2,3), inference fold(1,2,3,4,5)에 따라 환자별 db. gt_json 잡을 수 있도록 set up
    case = args.dataset # ['ROBOT', LAPA]
    anno_ver = '3'
    inference_fold = args.inference_fold
    save_path = args.inference_save_dir

    inference_assets = prepare_inference_aseets(case=case , anno_ver=anno_ver, inference_fold=inference_fold, save_path=save_path)
    patients = inference_assets['patients']
    
    patients_count = len(patients)

    # 2. for record metrics
    # ReportHelper
    patients_report_helper = ReportHelper(report_save_path=os.path.join(args.inference_save_dir, 'patients_report.csv'), report_type='patients')
    videos_report_helper = ReportHelper(report_save_path=os.path.join(args.inference_save_dir, 'videos_report.csv'), report_type='videos')

    patients_metrics_list = [] # save each patients metrics
    for idx in range(patients_count): # per patients
        patient = patients[idx]
        patient_no = patient['patient_no']
        patient_video = patient['patient_video']

        videos_metrics_list = [] # save each videos metrics

        # for save patients results
        each_patients_save_dir = os.path.join(args.inference_save_dir, patient_no)
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

            # metric per video
            video_metrics = evaluator.calc() # same return with metricHelper
            video_CR, video_OR = video_metrics['OOB_metric'], video_metrics['Over_estimation']
            video_TP, video_FP, video_TN, video_FN = video_metrics['TP'], video_metrics['FP'], video_metrics['TN'], video_metrics['FN']

            print('\t => video_name: {}'.format(video_name))
            print('\t    video_CR: {:.3f} | video_OR: {:.3f}'.format(video_CR, video_OR))
            print('\t    video_TP: {} | video_FP: {} | video_TN: {} | video_FN: {}'.format(video_TP, video_FP, video_TN, video_FN))

            # save video metrics
            video_report_col = videos_report_helper.get_report_form()

            video_report_col['Patient'] = patient_no
            video_report_col['Video'] = video_name
            video_report_col['FP'] = video_FP
            video_report_col['TP'] = video_TP
            video_report_col['FN'] = video_FN
            video_report_col['TN'] = video_TN
            video_report_col['TOTAL'] = video_FP + video_TP + video_FN + video_TN
            video_report_col['CR'] = video_CR
            video_report_col['OR'] = video_OR

            videos_report_helper.save_report(video_report_col)

            # for calc patients metric
            videos_metrics_list.append(video_metrics)
        
        # calc each patients CR, OR
        patient_metrics = MetricHelper().aggregate_calc_metric(videos_metrics_list)
        patient_CR, patient_OR = patient_metrics['OOB_metric'], patient_metrics['Over_estimation']
        patient_TP, patient_FP, patient_TN, patient_FN = patient_metrics['TP'], patient_metrics['FP'], patient_metrics['TN'], patient_metrics['FN']

        print('\t\t => patient_no: {}'.format(patient_no))
        print('\t\t    patient_CR: {:.3f} | patient_OR: {:.3f}'.format(patient_CR, patient_OR))
        print('\t\t    patient_TP: {} | patient_FP: {} | patient_TN: {} | patient_FN: {}'.format(patient_TP, patient_FP, patient_TN, patient_FN))

        # save patient metrics
        patients_report_col = patients_report_helper.get_report_form()

        patients_report_col['Patient'] = patient_no
        patients_report_col['FP'] = patient_FP
        patients_report_col['TP'] = patient_TP
        patients_report_col['FN'] = patient_FN
        patients_report_col['TN'] = patient_TN
        patients_report_col['TOTAL'] = patient_FP + patient_TP + patient_FN + patient_TN
        patients_report_col['CR'] = patient_CR
        patients_report_col['OR'] = patient_OR

        patients_report_helper.save_report(patients_report_col)
    
        # for calc total patients CR, OR
        patients_metrics_list.append(patient_metrics)

        # CLEAR PAGING CACHE
        clean_paging_chache()

    # for calc total patients CR, OR + (mCR, mOR)
    total_metrics = MetricHelper().aggregate_calc_metric(patients_metrics_list)
    total_mCR, total_mOR, total_CR, total_OR = total_metrics['mCR'], total_metrics['mOR'], total_metrics['OOB_metric'], total_metrics['Over_estimation']
            
    # return mCR, mOR, OR, CR of experiment
    return args, total_mCR, total_mOR, total_CR, total_OR
    
def main():    
    import os
    from core.utils.logging import ReportHelper # report helper (for experiments reuslts)

    # 0. set each experiment args 
    args = get_experiment_args()
    
    # 1. hyper prameter opts setup for experiments flow
    # 2. train
    args = train_main(args)

    # 3. inference
    args, experiment_mCR, experiment_mOR, experiment_CR, experiment_OR = inference_main(args)

    # 4. save experiments results
    experiments_results_path = '../results'
    os.makedirs(experiments_results_path, exist_ok=True)

    # using reportHelper
    patients_report_helper = ReportHelper(report_save_path=os.path.join(experiments_results_path, 'experiments_results.csv'), report_type='experiments')
    patients_report_col = patients_report_helper.get_report_form()

    patients_report_col['model'] = args.model
    patients_report_col['method'] = args.train_method
    patients_report_col['inference_fold'] = args.inference_fold
    patients_report_col['mCR'] = experiment_mCR
    patients_report_col['mOR'] = experiment_mOR
    patients_report_col['CR'] = experiment_CR
    patients_report_col['OR'] = experiment_OR

    patients_report_helper.save_report(patients_report_col)

if __name__ == '__main__':
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        print(base_path)

    main()






    