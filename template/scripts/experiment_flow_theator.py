def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    ### inference args
    parser.add_argument('--inference_save_dir', type=str, 
                        default='./resutls',
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
    args.model = 'resnet18'
    args.optimizer = 'sgd'
    args.lr_scheduler = 'mul_step_lr'
    # milestones = []
    # for it in range(1,7):
    #     for e in [9, 14]:
    #         milestones.append(e + (it-1) * 20)
    # print(milestones)
    # args.lr_milestones = [2, 4] #milestones

    ### dataset opts
    args.data_base_path = '/raid/img_db'
    args.batch_size = 64
    # args.IB_ratio = 7.7
    args.experiment_type = 'theator'

    ### train args
    args.save_path = '/OOB_RECOG/logs/theator'
    args.num_gpus = 1
    args.max_epoch = 120
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


def save_experiments(args):
    # TO DO : inference_main return으로 experiments 결과 저장, 저장하기 위해 experiemnts results sheet(csv) path가 args에 포함되어 있어야 하지 않을까?
    return 0


def train_main(args):
    print('train_main')
    
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.plugins import DDPPlugin

    from core.model import get_model, get_loss
    from core.api.theator_trainer import TheatorTrainer

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save_path,
        name='TB_log',
        default_hp_metric=False)

    x = TheatorTrainer(args)

    if args.num_gpus > 1:
        trainer = pl.Trainer(gpus=args.num_gpus, 
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,
                            plugins=DDPPlugin(find_unused_parameters=False), # [Warning DDP] error ?
                            accelerator='ddp')
    else:
        trainer = pl.Trainer(gpus=args.num_gpus,
                            # limit_train_batches=0.01,
                            # limit_val_batches=0.01,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)

    trainer.fit(x)
    
    _path = os.path.join(args.save_path, 'TB_log')
    args.restore_path = os.path.join(_path, os.listdir(_path)[-1]) # TO DO: we should define restore path
    # args.restore_path = os.path.join(x.restore_path)
    
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

    print('restore : ', args.restore_path)
    # from finetuning model
    model_path = get_inference_model_path(args.restore_path)
    model = CAMIO.load_from_checkpoint(model_path, args=args)
    model = model.cuda()

    # load inference dataset
    # TO DO : inference fold 에 따라 환자별 db. gt_json 잡을 수 있도록 set up (Inference, Evaluation module 사용시 for-loop로 set arguments)
    
    # use case 1 - init Inference
    # db_path = '/raid/OOB_Recog/img_db/ROBOT/R_100/01_G_01_R_100_ch1_01' # R_100_ch1_01
    db_path = '/raid/img_db/ROBOT/R_100/01_G_01_R_100_ch1_01' # R_100_ch1_01
    gt_json_path = '/nas2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_01_OOB_27.json'

    os.makedirs(args.inference_save_dir, exist_ok=True)
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
    args = get_experiment_args()
    
    # 1. hyper prameter opts setup for experiments flow
    # 2. train
    args = train_main(args)

    # 3. inference
    #args, mCR, mOR, CR, OR = inference_main(args)

    # 4. save experiments results [model, train_fold, inference_fold, ... , mCR, mOR, CR, OR]
    # save_experiments(args)

if __name__ == '__main__':
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        print(base_path)

    main()






    