import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin

from Model import CAMIO
from gen_dataset import CAMIO_Dataset
from gen_dataset import make_robot_csv

import pandas as pd
import shutil
from tqdm import tqdm

import time

import json

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='Size of the batches for each training step.')
    parser.add_argument('--max_epoch', type=int, help='The maximum number of training epoch.')
    parser.add_argument('--data_path', type=str, 
                        default='/data/CAM_IO/robot/images', help='Data path :)')
    parser.add_argument('--log_path', type=str, 
                        default='/OOB_RECOG/logs', help='log path :)')
    parser.add_argument('--num_gpus', type=int, 
                        default=2, help='The number of GPUs using for training.')

    ## init lr
    parser.add_argument('--init_lr', type=float, help='optimizer for init lr')

    ## training model
    parser.add_argument('--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d', 'mobilenet_v2', 'mobilenet_v3_small', 'squeezenet1_0'], help='backborn model')

    ## trian dataset video
    parser.add_argument('--train_videos', type=str, nargs='+',
                        default=['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R010', 'R013', 'R014', 'R015', 'R018', 
                            'R019', 'R048', 'R056', 'R074', 'R076', 'R084', 'R094', 'R100', 'R117', 'R201', 'R202', 'R203', 
                            'R204', 'R205', 'R206', 'R207', 'R209', 'R210', 'R301', 'R302', 'R304', 'R305', 'R313'],
                        choices=['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R010', 'R013', 'R014', 'R015', 'R017', 'R018', 
                                'R019', 'R022', 'R048', 'R056', 'R074', 'R076', 'R084', 'R094', 'R100', 'R116', 'R117', 'R201', 'R202', 'R203', 
                                'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R301', 'R302', 'R303', 'R304', 'R305', 'R313'], help='train video')

    ## val dataset video
    parser.add_argument('--val_videos', type=str, nargs='+',
                        default=['R017', 'R022', 'R116', 'R208', 'R303'],
                        choices=['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R010', 'R013', 'R014', 'R015', 'R017', 'R018', 
                                'R019', 'R022', 'R048', 'R056', 'R074', 'R076', 'R084', 'R094', 'R100', 'R116', 'R117', 'R201', 'R202', 'R203', 
                                'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R301', 'R302', 'R303', 'R304', 'R305', 'R313'], help='val video')

    ## random seed
    parser.add_argument('--random_seed', type=int, help='dataset ranbom seed')

    ## IB ratio
    parser.add_argument('--IB_ratio', type=int, help='IB = OOB * IB_ratio')

    ## log saved in project_name
    parser.add_argument('--project_name', type=str, help='log saved in project_name')

    ## Robot Lapa
    parser.add_argument('--dataset', type=str,
                        default='robot', choices=['robot', 'lapa'], help='[robot, lapa] choice on dataset')

    ## OOB NIR
    parser.add_argument('--task', type=str,
                        default='OOB', choices=['OOB', 'NIR'], help='[OOB, NIR] choice on task')



    args, _ = parser.parse_known_args()

    # hyper parameter setting
    # '__{}' parameters are not use in Model, just save record for train args parameter
    config_hparams = {
        '__project_name' : args.project_name,
        '__dataset' : args.dataset,
        '__task' : args.task,

        'backborn_model' : args.model,
        'optimizer_lr' : args.init_lr,

        '__max_epoch' : args.max_epoch,
        '__train_videos :' : args.train_videos,
        '__val_videos :' : args.val_videos,

        '__IB_ratio' : args.IB_ratio,
        '__random_seed' : args.random_seed
    }

    print('\n\n')
    print('batch size : ', args.batch_size)
    print('init lr : ', config_hparams['optimizer_lr'])
    print('backborn model : ', config_hparams['backborn_model'])
    print('\n\n')


    ### ### create results folder for save args and log.txt ### ###

    # log base path
    log_base_path = os.path.join(args.log_path, args.dataset, args.task)

    try :
        if not os.path.exists(os.path.join(log_base_path, args.project_name)):
            os.makedirs(os.path.join(log_base_path, args.project_name))
    except OSError :
        print('ERROR : Creating Directory, ' + os.path.join(log_base_path, args.project_name))

    # save args log
    log_txt='\n\n=============== \t\t COMMAND ARGUMENT \t\t ============= \n\n'
    log_txt+=json.dumps(args.__dict__, indent=2)

    # start time stamp
    startTime = time.time()
    s_tm = time.localtime(startTime)
    
    log_txt+='\n\n=============== \t\t TRAIN TIME \t\t ============= \n\n'
    log_txt+='STARTED AT : \t' + time.strftime('%Y-%m-%d %I:%M:%S %p \n', s_tm)
    
    save_log(log_txt, os.path.join(log_base_path, args.project_name, 'log.txt')) # save log

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    
    

    # 사용할 GPU 디바이스 번호들
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    # images path (.csv dir)
    base_path = args.data_path

    # bath size
    BATCH_SIZE = args.batch_size

    # make img info csv path
    # make_robot_csv(base_path, base_path)

    # model
    # model = CAMIO(config_hparams) # Trainer에서 사용할 모델 // add config_hparams 



    # dataset 설정
    # IB_ratio = [1,2,3,..] // IB개수 = OOB개수*IB_ratio
    trainset =  CAMIO_Dataset(csv_path=os.path.join(base_path, 'robot_oob_assets_path.csv'), patient_name=args.train_videos, is_train=True, random_seed=10, IB_ratio=args.IB_ratio)
    valiset =  CAMIO_Dataset(csv_path=os.path.join(base_path, 'robot_oob_assets_path.csv'), patient_name=args.val_videos, is_train=False, random_seed=10, IB_ratio=args.IB_ratio)
    

    print('trainset len : ', len(trainset))
    print('valiset len : ', len(valiset))

    exit(0)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
                                            shuffle=False, num_workers=8)
    vali_loader = torch.utils.data.DataLoader(valiset, batch_size=BATCH_SIZE, 
                                            shuffle=False, num_workers=8)

    
    
    ###### this section is for check that dataset gets img and label correctly ####
    ### 해당 section은 단지 모델이 학습하는 데이터셋이 무엇인지 확인하기 위해 구성된 Dataset의 image와 label정보를 모두 /get_dataset_results에 저장하는 부분
    ### 학습시 꼭 필요한 부분이 아님.
    # datset log check
    ''' 
    trainset_save_path = os.path.join('get_dataset_results', 'trainset')
    valiset_save_path = os.path.join('get_dataset_results', 'valiset')

    trainset_df = pd.DataFrame({
        'img_list' : trainset.img_list,
        'label_list' : trainset.label_list
    })

    valiset_df = pd.DataFrame({
        'img_list' : valiset.img_list,
        'label_list' : valiset.label_list
    })
    
    # csv 저장
    # trainset_df.to_csv(os.path.join(trainset_save_path, 'trainset.csv'), mode='w')
    valiset_df.to_csv(os.path.join(valiset_save_path, 'valiset.csv'), mode='w')

    # 사용된 이미지(img_list) 파일복사 | train

    label_class = [0, 1]
    copied_count = [0, 0]
    for tar_label in label_class :
        for img_path in tqdm(trainset_df[trainset_df['label_list'] == tar_label]['img_list'], desc='trainset coping_{}'.format(tar_label)) :
            file_name, file_ext = os.path.basename(img_path).split('.')
            copy_name = file_name + '_copy_' + str(tar_label) + '.' + file_ext
            shutil.copyfile(img_path, os.path.join(trainset_save_path, str(tar_label), copy_name))

            copied_count[tar_label]+=1
    

    print('train copeied count : ', copied_count)
    

    # 사용된 이미지(img_list) 파일복사 | validation
    label_class = [0, 1]
    copied_count = [0, 0]
    for tar_label in label_class :
        for img_path in tqdm(valiset_df[valiset_df['label_list'] == tar_label]['img_list'], desc='valiset coping_{}'.format(tar_label)) :
            file_name, file_ext = os.path.basename(img_path).split('.')
            copy_name = file_name + 'copy' + '_' + str(tar_label) + '.' + file_ext
            shutil.copyfile(img_path, os.path.join(valiset_save_path, str(tar_label), copy_name))

            copied_count[tar_label]+=1

    print('vali copeied count : ', copied_count)
    '''

    ##### ### ###

    """
        dirpath : log 저장되는 위치
        filename : 저장되는 checkpoint file 이름
        monitor : 저장할 metric 기준
    """
    checkpoint_filename = 'ckpoint_{}-model={}-batch={}-lr={}-'.format(args.project_name, args.model, BATCH_SIZE, args.init_lr)
    checkpoint_filename = checkpoint_filename + '{epoch}-{val_loss:.4f}'
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_base_path, args.project_name), filename=checkpoint_filename, # {epoch}-{val_loss:.4f}
            save_top_k=1, save_last=True, verbose=True, monitor="val_loss", mode="min"
    )

    # change last checkpoint name
    checkpoint_callback.CHECKPOINT_NAME_LAST = 'ckpoint_{}-model={}-batch={}-lr={}-'.format(args.project_name, args.model, BATCH_SIZE, args.init_lr) + '{epoch}-last'

    """
        tensorboard logger
        save_dir : checkpoint log 저장 위치처럼 tensorboard log 저장위치
        name : tensorboard log 저장할 폴더 이름 (이 안에 하위폴더로 version_0, version_1, ... 이런식으로 생김)
        default_hp_metric : 뭔지 모르는데 거슬려서 False
    """
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_base_path, args.project_name),
                                            name='TB_log',
                                            default_hp_metric=False)
    """
        gpus : GPU 몇 개 사용할건지, 1개면 ddp 안함
        max_epochs : 최대 몇 epoch 할지
        checkpoint_callback : 위에 정의한 callback 함수
        logger : tensorboard logger, 다른 custom logger도 사용가능
        accelerator : 멀티 GPU 모드 설정
    """

    # pytorch lightning Trainer Class
    ## train    
    trainer = pl.Trainer(gpus=args.num_gpus, 
                        max_epochs=args.max_epoch, 
                        checkpoint_callback=checkpoint_callback,
                        logger=tb_logger,
                        plugins=DDPPlugin(find_unused_parameters=False), # [Warning DDP] error ?
                        accelerator='ddp')


    trainer.fit(model, train_loader, vali_loader)


    # test
    # model = model.load_from_checkpoint('/home/mkchoi/logs/OOB_robot_test/epoch=0-val_loss=0.2456.ckpt')
    # trainer.test(model, test_loader)

    # finish time stamp
    finishTime = time.time()
    f_tm = time.localtime(finishTime)

    log_txt = 'FINISHED AT : \t' + time.strftime('%Y-%m-%d %I:%M:%S %p \n', f_tm)
    save_log(log_txt, os.path.join(log_base_path, args.project_name, 'log.txt')) # save log



# save log 
def save_log(log_txt, save_dir) :
    print('=========> SAVING LOG ... | {}'.format(save_dir))
    with open(save_dir, 'a') as f :
        f.write(log_txt)




if __name__ == "__main__":
    train()