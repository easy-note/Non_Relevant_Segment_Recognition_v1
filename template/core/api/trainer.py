import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.api import BaseTrainer
from core.model import get_model, get_loss
from core.dataset import *
from core.utils.metric import MetricHelper



class CAMIO(BaseTrainer):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
        # TODO 필요한거 더 추가하기
        self.args = args

        self.model = get_model(self.args)
        self.loss_fn = get_loss(self.args)

        self.metric_helper = MetricHelper()
        self.hem_helper = HEMHelper()

        # only use for HEM
        self.train_method = self.args.train_method
        self.hem_helper.set_method(self.train_method)

        if self.train_method in ['hem-softmax', 'hem-vi']:
            self.reset_epoch = self.args.max_epoch // 2 - 1
        else:
            self.reset_epoch = -1


    def setup(self, stage):
        # training stage
        if stage == 'fit' or stage is None:
            if self.args.dataset == 'ROBOT':
                self.trainset = RobotDataset(self.args, state='train')
                self.valset = RobotDataset(self.args, state='val')
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
        return DataLoader(
            self.trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.num_workers,
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
        if self.train_method == 'hem-bs':
            # TODO future works
            pass
            # if self.training:
            #     ids = self.hem_helper(self.model, self.train_loader)
            #     self.trainset.set_sample_ids(ids)

        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        """
            forward for mini-batch
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
        }

    def validation_epoch_end(self, outputs):
        self.metric_helper.calc_metric()
        
        for k, v in metrics.items():
            if k in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                self.log('val_'+k, v, on_epoch=True, prog_bar=True)

        if not self.args.use_lightinig_style_save:
            self.save_checkpoint()

        # Hard Example Mining (Offline)
        if self.current_epoch == self.reset_epoch and self.train_method in ['hem-softmax', 'hem-vi']:
            self.trainset.change_mode(True)
            self.valset.change_mode(True)
            
            hem_train_ids = self.hem_helper(self.model, self.train_loader)
            hem_val_ids = self.hem_helper(self.model, self.train_loader)
            self.trainset.set_sample_ids(hem_train_ids)
            self.valset.set_sample_ids(hem_val_ids)
            
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu())

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'test_loss': loss,
        }

    def test_epoch_end(self, outputs):
        self.metric_helper.calc_metric()
        
        for k, v in metrics.items():
            if k in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                self.log('test_'+k, v, on_epoch=True, prog_bar=True)

    def online_hem(self):
        pass