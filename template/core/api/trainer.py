import os
import math
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from core.api import BaseTrainer
from core.model import get_model, get_loss
from core.dataset import *
from core.utils.metric import MetricHelper


import csv


class CAMIO(BaseTrainer):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
 
        # TODO 필요한거 더 추가하기
        self.args = args

        random.seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        os.environ["PYTHONHASHSEED"]=str(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed(self.args.random_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=True
        
        
        self.save_hyperparameters() # save with hparams

        self.model = get_model(self.args)
        self.loss_fn = get_loss(self.args)

        self.metric_helper = MetricHelper()
        self.hem_helper = HEMHelper(self.args)

        self.best_val_loss = math.inf

        self.sanity_check = True
        self.restore_path = None # inference module args / save path of hem df 
        
        self.last_epoch = -1
        self.hem_extract_mode = self.args.hem_extract_mode

        # hem-online
        if 'online' in self.args.hem_extract_mode:
            self.hem_helper.set_method(self.hem_extract_mode)
        
        # hem-offline // hem_train, general_train 에서는 Hem 생성하지 않음.
        elif 'offline' in self.args.hem_extract_mode:
            self.hem_helper.set_method(self.hem_extract_mode)            
            self.last_epoch = self.args.max_epoch - 1

    def setup(self, stage):
        '''
            Called one each GPU separetely - stage defines if we are at fit or test step.
            We wet up only relevant datasets when stage is specified (automatically set by pytorch-lightning).
        '''
        # training stage
        if stage == 'fit' or stage is None:
            if self.args.dataset == 'ROBOT':
                self.trainset = RobotDataset(self.args, state='train') # train dataset setting
                self.valset = RobotDataset(self.args, state='val') # val dataset setting
            elif self.args.dataset == 'LAPA':
                self.trainset = LapaDataset(self.args, state='train') 
                self.valset = LapaDataset(self.args, state='val')

        # testing stage
        if stage in (None, 'test'):
            if self.args.dataset == 'ROBOT':
                self.testset = RobotDataset(self.args, state='val')
            elif self.args.dataset == 'LAPA':
                self.testset = LapaDataset(self.args, state='val')

    def train_dataloader(self):
        if 'hem-focus' in self.hem_extract_mode:
            print(len(self.trainset)//self.args.batch_size,)
            
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                sampler=FocusSampler(self.trainset.label_list,
                                     self.args)
            )
        else:
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
                drop_last=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        # TODO testset이 따로 있으면 그걸로 하기
        return DataLoader(
            self.testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )

    def forward(self, x):
        """
            forward data
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if 'hem-emb' in self.hem_extract_mode and self.training:
            img_path, x, y = batch
            loss = self.hem_helper.compute_hem(self.model, x, y, self.loss_fn)
            
        elif 'hem-focus' in self.hem_extract_mode and self.training:
            img_path, x, y = batch
            
            loss = self.hem_helper.hem_cos_hard_sim(self.model, x, y, self.loss_fn)
            
        else:
            img_path, x, y = batch

            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return  {
            'loss': loss,
        }

    def training_epoch_end(self, outputs):
        train_loss, cnt = 0, 0

        for output in outputs:
            train_loss += output['loss'].cpu().data.numpy()
            cnt += 1

        train_loss_mean = train_loss/cnt

        # write train loss
        self.metric_helper.write_loss(train_loss_mean, task='train')

    def validation_step(self, batch, batch_idx): # val - every batch
        img_path, x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu()) # MetricHelper 에 저장

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
            'img_path': img_path,
            'x': x.detach().cpu(),
            'y': y.detach().cpu(),
            'y_hat': y_hat.argmax(dim=1).detach().cpu(),
            'logit': y_hat.detach().cpu()
        }

    def validation_epoch_end(self, outputs): # val - every epoch
        if self.sanity_check:
            print('sanity check')
            self.restore_path = os.path.join(self.args.save_path, self.logger.log_dir) # hem-df path / inference module restore path
            print('SANITY RESTORE PATH : ', self.restore_path)

            self.sanity_check = False

        else:
            self.restore_path = os.path.join(self.args.save_path, self.logger.log_dir) # hem-df path / inference module restore path
            
            metrics = self.metric_helper.calc_metric() # 매 epoch 마다 metric 계산 (TP, TN, .. , accuracy, precision, recaull, f1-score)
        
            val_loss, cnt = 0, 0
            for output in outputs: 
                val_loss += output['val_loss'].cpu().data.numpy()
                cnt += 1

            val_loss_mean = val_loss/cnt
            metrics['Loss'] = val_loss_mean

            '''
                metrics = {
                    'TP': cm.TP[self.OOB_CLASS],
                    'TN': cm.TN[self.OOB_CLASS],
                    'FP': cm.FP[self.OOB_CLASS],
                    'FN': cm.FN[self.OOB_CLASS],
                    'Accuracy': cm.ACC[self.OOB_CLASS],
                    'Precision': cm.PPV[self.OOB_CLASS],
                    'Recall': cm.TPR[self.OOB_CLASS],
                    'F1-Score': cm.F1[self.OOB_CLASS],
                    'OOB_metric': (metrics['TP']-metrics['FP']) / (metrics['FN'] + metrics['TP'] + metrics['FP']) # 잘못예측한 OOB / predict OOB + 실제 OOB
                    'Over_estimation': metrics['FP'] / (metrics['FN'] + metrics['TP'] + metrics['FP']) # OR
                    'Under_estimation': metrics['FN'] / (metrics['FN'] + metrics['TP'] + metrics['FP'])
                    'Correspondence_estimation': metrics['TP'] / (metrics['FN'] + metrics['TP'] + metrics['FP']) # CR
                    'UNCorrespondence_estimation': (metrics['FP'] + metrics['FN']) / (metrics['FN'] + metrics['TP'] + metrics['FP'])
                    'Mean_metric': (metrics['Correspondence_estimation'] + (1-metrics['Over_estimation'])) / 2.
                    'Loss': val_loss/cnt
                }
            '''

            self.log_dict(metrics, on_epoch=True, prog_bar=True)
            
            # save result.csv 
            self.metric_helper.save_metric(metric=metrics, epoch=self.current_epoch, args=self.args, save_path=os.path.join(self.args.save_path, self.logger.log_dir))

            # write val loss
            self.metric_helper.write_loss(val_loss_mean, task='val')
            self.metric_helper.save_loss_pic(save_path=os.path.join(self.args.save_path, self.logger.log_dir))

            if not self.args.use_lightning_style_save:
                if self.best_val_loss > val_loss_mean : # math.inf 보다 현재 epoch val loss 가 작으면,
                    self.best_val_loss = val_loss_mean # self.best_val_loss 업데이트. 
                    self.save_checkpoint()

                if self.current_epoch + 1 == self.args.max_epoch: # max_epoch 모델 저장
                    # TODO early stopping 적용시 구현 필요
                    self.best_val_loss = val_loss_mean
                    self.save_checkpoint()

            # Hard Example Mining (Offline)
            if self.current_epoch == self.last_epoch:
                if self.args.stage not in ['hem_train', 'general_train'] and 'offline' in self.args.hem_extract_mode: 
                    hem_df, total_dataset_len_list, hem_dataset_len_list = self.hem_helper.compute_hem(self.model, outputs)
                    hem_df.to_csv(os.path.join(self.restore_path, '{}-{}-{}.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)

                    # field names  
                    cols = ['hard_neg_df', 'hard_pos_df', 'vanila_neg_df', 'vanila_pos_df']  
                    with open(os.path.join(self.restore_path, 'DATASET_COUNT.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), 'w',newline='') as f: 
                        write = csv.writer(f) 
                        write.writerow(cols) 
                        write.writerow(total_dataset_len_list) 
                        write.writerow(hem_dataset_len_list)


    def test_step(self, batch, batch_idx):
        img_path, x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu())

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'test_loss': loss,
        }

    def test_epoch_end(self, outputs):
        metrics = self.metric_helper.calc_metric()
        
        for k, v in metrics.items():
            if k in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                self.log('test_'+k, v, on_epoch=True, prog_bar=True)
