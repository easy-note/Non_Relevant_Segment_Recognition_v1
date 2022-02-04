import os
import math
import numpy as np
import csv
import json
import copy
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.api import BaseTrainer
from core.model import get_model, get_loss
from core.dataset import *
from core.utils.metric import MetricHelper
from core.utils.misc import save_dict_to_csv, save_dataset_info
# save_dict_to_csv: results 저장시 사용, 사용안하고 싶으면 한줄만 빼기 (not core function)
# save_dataset_info: trainer 에서 사용한 train, val set 정보 저장



class CAMIO(BaseTrainer):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
 
        # TODO 필요한거 더 추가하기
        self.args = args
        if hasattr(self.args, 'cur_stage'):
            self.cur_stage = self.args.cur_stage
        self.save_hyperparameters() # save with hparams
        
        self.model = get_model(self.args)
        self.loss_fn = get_loss(self.args)

        self.metric_helper = MetricHelper()
        self.hem_helper = HEMHelper(self.args)

        self.best_val_loss = math.inf
        self.best_mean_metric = -2 # repvgg use only

        self.sanity_check = True
        self.restore_path = self.args.restore_path # inference module args / save path of hem df 
        
        self.cur_step = 0
        self.last_epoch = -1
        self.hem_extract_mode = self.args.hem_extract_mode

        # hem-online
        if 'online' in self.args.hem_extract_mode:
            self.hem_helper.set_method(self.hem_extract_mode)
        
        # hem-offline // hem_train, general_train 에서는 Hem 생성하지 않음.
        elif 'offline' in self.args.hem_extract_mode:
            self.hem_helper.set_method(self.hem_extract_mode)
            self.last_epoch = self.args.max_epoch - 1

        self.emb_only = self.args.use_emb_only

    def on_epoch_start(self):
        print('ON EPOCH START')
        print(self.restore_path, hasattr(self, 'cur_stage'))
        if hasattr(self, 'cur_stage'):
            print('restore path : ', self.restore_path)
            
            # if self.training and self.restore_path is not None and self.current_epoch == 0:
            if self.cur_stage > 1 and self.restore_path is not None and self.current_epoch == 0:
                trainset = copy.deepcopy(self.trainset)
                trainset.img_list = trainset.img_list[trainset.split_patient:]
                trainset.label_list = trainset.label_list[trainset.split_patient:]
                
                d_loader = DataLoader(trainset, 
                                    batch_size=self.args.batch_size, 
                                    shuffle=False, 
                                    num_workers=self.args.num_workers)
                
                change_list = []
                
                self.model.eval()
                with torch.no_grad():
                    for _, img, lbs in tqdm(d_loader):
                        img = img.cuda()
                        outputs = self.model(img)
                        ids = list(torch.argmax(outputs, -1).cpu().data.numpy())
                        change_list += ids
                
                self.trainset.label_list[self.trainset.split_patient:] = change_list
                
                self.args.restore_path = None
                self.model = get_model(self.args).cuda()
        
    def save_trainer_dataset_info(self): # 해당 trainer에서 사용한 train dataset, val dataset 정보 대해서 저장
        train_dataset_info_path = os.path.join(self.restore_path, 'train_dataset_info.json')
        save_dataset_info(self.trainset, train_dataset_info_path)

        val_dataset_info_path = os.path.join(self.restore_path, 'val_dataset_info.json')
        save_dataset_info(self.valset, val_dataset_info_path)

    def setup(self, stage): # train할때만 setup, model 불러올때는 setup x
        '''
            Called one each GPU separetely - stage defines if we are at fit or test step.
            We wet up only relevant datasets when stage is specified (automatically set by pytorch-lightning).
        '''
        # training stage
        if stage == 'fit' or stage is None:
            if self.args.dataset == 'ROBOT':
                if self.args.train_stage == 'general_train': # original (20.80 데이터 사용)
                    self.trainset = RobotDataset_new(self.args, state='train', wise_sample=self.args.use_wise_sample) # train dataset setting
                    self.valset = RobotDataset_new(self.args, state='val') # val dataset setting
                
                elif self.args.train_stage == 'hem_train': # offline (apply) => load from appointmnet assets (== 뽑힌 hem assets)
                    if 'offline' in self.args.hem_extract_mode:
                        self.trainset = RobotDataset_new(self.args, state='train', appointment_assets_path=self.args.appointment_assets_path) # load from hem_assets
                        self.valset = RobotDataset_new(self.args, state='val')
                        
                    elif 'online' in self.args.hem_extract_mode: # online이 여기로 들어가면 hem_train 의미상 맞을듯.. // 기존에 subset 불러왔으니 맞을듯..
                        self.trainset = RobotDataset_new(self.args, state='train', wise_sample=self.args.use_wise_sample) # train dataset setting
                        self.valset = RobotDataset_new(self.args, state='val') # val dataset setting
                
                else : # mini_fold 1 2 3 4 ==> general (60개 환자만 train / 20개 환자 validation) baby model train stage
                    train_stage_to_minifold = {
                        'mini_fold_stage_0': '1',
                        'mini_fold_stage_1': '2',
                        'mini_fold_stage_2': '3',
                        'mini_fold_stage_3': '4',
                    }

                    if self.args.hem_interation_idx == 100: # 초기 hem iteration => load from sub/meta set
                        self.trainset = RobotDataset_new(self.args, state='train_mini', minifold=train_stage_to_minifold[self.args.train_stage], wise_sample=self.args.use_wise_sample) # train dataset setting
                        self.valset = RobotDataset_new(self.args, state='val_mini', minifold=train_stage_to_minifold[self.args.train_stage]) # val dataset setting
                    
                    else: # 200, 300 ==> load from appointment assets (== 뽑힌 hem assets)
                        self.trainset = RobotDataset_new(self.args, state='train_mini', minifold=train_stage_to_minifold[self.args.train_stage], appointment_assets_path=self.args.appointment_assets_path) # load from hem_assets
                        self.valset = RobotDataset_new(self.args, state='val_mini', minifold=train_stage_to_minifold[self.args.train_stage], appointment_assets_path=self.args.appointment_assets_path)
            
            elif self.args.dataset == 'LAPA':
                self.trainset = LapaDataset(self.args, state='train') 
                self.valset = LapaDataset(self.args, state='val')

        # testing stage
        if stage in (None, 'test'):
            if self.args.dataset == 'ROBOT':
                self.testset = RobotDataset_new(self.args, state='val')
            elif self.args.dataset == 'LAPA':
                self.testset = LapaDataset(self.args, state='val')

    def train_dataloader(self):
        if 'hem-focus' in self.hem_extract_mode:
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                sampler=FocusSampler(self.trainset.label_list,
                                     self.args)
            )
        elif self.args.experiment_type == 'theator':
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                sampler=MPerClassSampler(self.trainset.label_list, 
                                        self.args.batch_size//2, self.args.batch_size)
            )
        else:
            if self.args.use_oversample:
                return DataLoader(
                    self.trainset,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    sampler=MPerClassSampler(self.trainset.label_list, 
                                            self.args.batch_size//2, self.args.batch_size)
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
        if self.emb_only:
            emb = self.model(x)
            
            sim_dist = torch.zeros((emb.size(0), self.model.proxies.size(1))).to(emb.device)
        
            for d in range(sim_dist.size(1)):
                sim_dist[:, d] = 1 - torch.nn.functional.cosine_similarity(emb, self.model.proxies[:, d].unsqueeze(0))
                
            return sim_dist
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):

        if 'hem-emb' in self.hem_extract_mode and self.training:
            img_path, x, y = batch
            loss = self.hem_helper.compute_hem(self.model, x, y, self.loss_fn)
            
        elif 'hem-focus' in self.hem_extract_mode and self.training:
            img_path, x, y = batch
            
            loss = self.hem_helper.compute_hem(self.model, x, y, self.loss_fn)
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

        # save number of train dataset (rs, nrs)
        if self.current_epoch == self.last_epoch:
            pass

    def validation_step(self, batch, batch_idx): # val - every batch
        img_path, x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu()) # MetricHelper 에 저장

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
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

            self.log_dict(metrics, on_epoch=True, prog_bar=True)
            
            # save result.csv 
            metrics_save_data = dict({'Model': self.args.model, 'Epoch': self.current_epoch}, **metrics)

            save_dict_to_csv(metrics_save_data, os.path.join(self.args.save_path, self.logger.log_dir, 'train_metric.csv'))
            # self.metric_helper.save_metric(model_name=args.model_name, metric=metrics, epoch=self.current_epoch, args=self.args, save_path=os.path.join(self.args.save_path, self.logger.log_dir))

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

            if self.current_epoch == self.last_epoch: # last epoch => save dataset info
                self.save_trainer_dataset_info()
                        
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
