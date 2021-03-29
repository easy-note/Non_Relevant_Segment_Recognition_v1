import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from Model import CAMIO
from gen_dataset import CAMIO_Dataset

import time

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, 
                        default=16, help='Size of the batches for each training step.')
    parser.add_argument('--max_epoch', type=int, 
                        default=20, help='The maximum number of training epoch.')
    parser.add_argument('--data_path', type=str, 
                        default='/data/CAM_IO/robot/images', help='Data path :)')
    parser.add_argument('--log_path', type=str, 
                        default='/CAM_IO/logs', help='log path :)')
    parser.add_argument('--num_gpus', type=int, 
                        default=2, help='The number of GPUs using for training.')

    ## log saved in project_name
    parser.add_argument('--project_name', type=str, 
                        default='robot_oob_train_2', help='log saved in project_name')

    ## Robot Lapa
    parser.add_argument('--dataset', type=str, 
                        default='robot', choices=['robot', 'lapa'], help='[robot, lapa] choice on dataset')

    ## OOB NIR
    parser.add_argument('--task', type=str, 
                        default='OOB', choices=['OOB', 'NIR'], help='[OOB, NIR] choice on task')


    args, _ = parser.parse_known_args()


    # 사용할 GPU 디바이스 번호들
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    # extracted images path
    base_path = args.data_path
    
    # log base path
    log_base_path = os.path.join(args.log_path, args.dataset, args.task)

    # bath size
    BATCH_SIZE = args.batch_size

    # model
    model = CAMIO() # Trainer에서 사용할 모델

    # dataset 설정
    trainset =  CAMIO_Dataset(base_path, is_train=True, test_mode=False, data_ratio=0.2)
    valiset = CAMIO_Dataset(base_path, is_train=False, test_mode=False, data_ratio=0.2)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
                                            shuffle=True, num_workers=2)
    vali_loader = torch.utils.data.DataLoader(valiset, batch_size=BATCH_SIZE, 
                                            shuffle=False, num_workers=2)

    """
        dirpath : log 저장되는 위치
        filename : 저장되는 checkpoint file 이름
        monitor : 저장할 metric 기준
    """
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_base_path, args.project_name), filename='{epoch}-{val_loss:.4f}',
            save_top_k=1, save_last=True, verbose=True, monitor="val_loss", mode="min",
        )

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
                        accelerator='ddp')
    trainer.fit(model, train_loader, vali_loader)


    # test
    # model = model.load_from_checkpoint('/home/mkchoi/logs/OOB_robot_test/epoch=0-val_loss=0.2456.ckpt')
    # trainer.test(model, test_loader)

if __name__ == "__main__":
    train()