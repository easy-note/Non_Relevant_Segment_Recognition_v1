def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    ### inference args
    parser.add_argument('--inference_save_dir', type=str, 
                        default='../restuls',
                        help='root directory for infernce saving')
        

    parser.add_argument('--inference_interval', type=int, 
                        default=30,
                        help='Inference Interval of frame')

    parser.add_argument('--inference_fold',
                    default='3',
                    type=str,
                    choices=['1', '2', '3', '4', '5', 'free'],
                    help='valset 1, 2, 3, 4, 5, free=for setting train_videos, val_vidoes')


    args = parser.parse_args()

    ### model basic info opts
    args.model = 'mobilenet_v3_large'

    ### dataset opts
    args.data_base_path = '/raid/img_db'
    
    ### train args
    args.save_path = '/OOB_RECOG/logs/project-1'
    args.num_gpus = 1
    args.max_epoch = 2
    args.min_epoch = 1

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


def save_experiments(args):
    # TO DO : inference_main return으로 experiments 결과 저장, 저장하기 위해 experiemnts results sheet(csv) path가 args에 포함되어 있어야 하지 않을까?
    return 0


def train_main(args):
    print('train_main')
    
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.plugins import DDPPlugin

    from core.model import get_model, get_loss
    from core.api.trainer import CAMIO

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
        trainer = pl.Trainer(gpus=args.num_gpus,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)

    trainer.fit(x)

    args.restore_path = os.path.join(args.save_path, 'TB_log', 'version_4') # TO DO: we should define restore path
    
    return args


def inference_main(args):
    print('inference_main')
    
    ### test inference module
    from core.api.trainer import CAMIO
    from core.api.inference import InferenceDB # inference module
    from core.api.evaluation import Evaluator # evaluation module

    import os
    import pandas as pd
    import glob


    # from pretrained model
    '''
    model = CAMIO(args)
    model = model.cuda()
    '''

    # from finetuning model
    model_path = get_inference_model_path(args.restore_path)
    model = CAMIO.load_from_checkpoint(model_path, args=args)
    model = model.cuda()

    # load inference dataset
    # TO DO : inference fold 에 따라 환자별 db. gt_json 잡을 수 있도록 set up (Inference, Evaluation module 사용시 for-loop로 set arguments)
    
    # use case 1 - init Inference
    db_path = '/raid/img_db/ROBOT/R_100/01_G_01_R_100_ch1_01' # R_100_ch1_01
    gt_json_path = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_01_OOB_27.json'
    predict_csv_path = os.path.join(args.inference_save_dir, 'R_100_ch1_01.csv')
    metric_path = os.path.join(args.inference_save_dir, 'R_100_ch1_01-metric.json')
    
    # Inference module
    inference = InferenceDB(model, db_path, args.inference_interval) # Inference object
    predict_list, target_img_list, target_frame_idx_list = inference.start() # call start

    # save predict list to csv
    predict_df = pd.DataFrame({
                    'frame_idx': target_frame_idx_list,
                    'predict': predict_list,
                    'target_img': target_img_list,
                })
    predict_df.to_csv(predict_csv_path)

    # Evaluation module
    evaluator = Evaluator(predict_csv_path, gt_json_path, args.inference_interval)
    metrics = evaluator.calc() # same return with metricHelper
    CR, OR = metrics['OOB_metric'], metrics['Over_estimation']
    TP, FP, TN, FN = metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
    
    print(CR, OR)
    print(TP, FP, TN, FN)

    # return mCR, mOR
    # TO DO : Inference 완료시 Pateints mCR, mOR, CR, OR 기록을 위해 return 필요
    mCR, mOR, CR, OR = 0,0,0,0
    return args, mCR, mOR, CR, OR
    
    
def main():    

    # 0. set each experiment args 
    args_list = [get_experiment_args(), get_experiment_args()] # 상황에 따라 다르게 args 구성
    
    # 1. hyper prameter opts setup for experiments flow
    for args in args_list:
        # 2. train
        args = train_main(args)

        # 3. inference
        args, mCR, mOR, CR, OR = inference_main(args)

        # 4. save experiments results [model, train_fold, inference_fold, ... , mCR, mOR, CR, OR]
        # save_experiments(args)

if __name__ == '__main__':
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        print(base_path)

    main()






    