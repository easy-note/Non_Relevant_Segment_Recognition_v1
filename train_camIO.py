import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from Model import CAMIO
from gen_dataset import CAMIO_Dataset


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, 
                        default=64, help='Size of the batches for each training step.')
    parser.add_argument('--max_epoch', type=int, 
                        default=20, help='The maximum number of training epoch.')
    parser.add_argument('--data_path', type=str, 
                        default='/data/CAM_IO', help='Data path :)')
    parser.add_argument('--log_path', type=str, 
                        default='/CAM_IO', help='Data path :)')
    parser.add_argument('--num_gpus', type=int, 
                        default=2, help='The number of GPUs using for training.')

    args, _ = parser.parse_known_args()


    # 사용할 GPU 디바이스 번호들
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    # extracted images path
    base_path = args.data_path
    dpath = args.log_path

    BATCH_SIZE = args.batch_size
    model = CAMIO()

    trainset =  CAMIO_Dataset(base_path, is_train=True, data_ratio=0.5)
    testset = CAMIO_Dataset(base_path, is_train=False, data_ratio=0.5)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
                                            shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, 
                                            shuffle=False, num_workers=8)

    """
        dirpath : log 저장되는 위치
        filename : 저장되는 checkpoint file 이름
        monitor : 저장할 metric 기준
    """
    checkpoint_callback = ModelCheckpoint(
            dirpath=dpath + '/logs/OOB_robot_test', filename='{epoch}-{val_loss:.4f}',
            save_top_k=1, save_last=True, verbose=True, monitor="val_loss", mode="min",
        )

    """
        tensorboard logger
        save_dir : checkpoint log 저장 위치처럼 tensorboard log 저장위치
        name : tensorboard log 저장할 폴더 이름 (이 안에 하위폴더로 version_0, version_1, ... 이런식으로 생김)
        default_hp_metric : 뭔지 모르는데 거슬려서 False
    """
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dpath + '/logs/OOB_robot_test',
                                            name='DPP_Test',
                                            default_hp_metric=False)
    """
        gpus : GPU 몇 개 사용할건지, 1개면 ddp 안함
        max_epochs : 최대 몇 epoch 할지
        checkpoint_callback : 위에 정의한 callback 함수
        logger : tensorboard logger, 다른 custom logger도 사용가능
        accelerator : 멀티 GPU 모드 설정
    """
    trainer = pl.Trainer(gpus=args.num_gpus, 
                        max_epochs=args.max_epoch, 
                        checkpoint_callback=checkpoint_callback,
                        logger=tb_logger,
                        accelerator='ddp')
    trainer.fit(model, train_loader, test_loader)

# model = model.load_from_checkpoint('/home/mkchoi/logs/OOB_robot_test/epoch=0-val_loss=0.2456.ckpt')
# trainer.test(model, test_loader)

if __name__ == "__main__":
    train()