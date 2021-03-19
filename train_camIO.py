import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from Model import CAMIO
from gen_dataset import CAMIO_Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# original video anno path
# video_path = '/home/mkchoi/dataset/CAM'

# extracted images path
base_path = '/home2/camIO'
dpath = '/home/mkchoi/camIO_lightning'

BATCH_SIZE = 64
model = CAMIO()

trainset =  CAMIO_Dataset(base_path, True, 0.5)
testset = CAMIO_Dataset(base_path, False, 0.5)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

checkpoint_callback = ModelCheckpoint(
        dirpath=dpath + '/logs/OOB_robot_test', filename='{epoch}-{val_loss:.4f}',
        save_top_k=1, save_last=True, verbose=True, monitor="val_loss", mode="min",
    )

tb_logger = pl_loggers.TensorBoardLogger(save_dir=dpath + '/logs/OOB_robot_test',
                                        name='DPP_Test',
                                        default_hp_metric=False)
trainer = pl.Trainer(gpus=2, 
                    max_epochs=20, 
                    checkpoint_callback=checkpoint_callback,
                    logger=tb_logger,
                    accelerator='ddp')
trainer.fit(model, train_loader, test_loader)

# model = model.load_from_checkpoint('/home/mkchoi/logs/OOB_robot_test/epoch=0-val_loss=0.2456.ckpt')
# trainer.test(model, test_loader)
