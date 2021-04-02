import torch
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.metrics.utils import to_categorical
from pytorch_lightning.metrics import Accuracy, Precision, Recall, ConfusionMatrix, F1
from pycm import *

### hparamds
'''
{
    'lr' : 1e-4,

}
'''

class CAMIO(pl.LightningModule):
    def __init__(self, hparams:dict):
        super().__init__()
        # self.model = models.resnet50(pretrained=True)
        self.model = models.wide_resnet50_2(pretrained=True)
        # self.model = models.densenet201(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        # self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 2)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.softmax = torch.nn.Softmax()
        
        # hyper param setting
        self.hparmas = hparams
        self.init_lr = hparams['optimizer_lr']

        print(hparams)
        
        self.accuracy = Accuracy()
        self.prec = Precision(num_classes=1, is_multiclass=False)
        self.rc = Recall(num_classes=1, is_multiclass=False)
        self.f1 = F1(num_classes=1, multilabel=False)
        # self.confmat = ConfusionMatrix(num_classes=1)

        self.preds = []
        self.gts = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx): # 배치마다 실행
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        c_hat = to_categorical(y_hat)
        acc = self.accuracy(c_hat, y)
        prec = self.prec(c_hat, y)
        rc = self.rc(c_hat, y)
        f1 = self.f1(c_hat, y)

        for _y, _y_hat in zip(y, c_hat):
            self.preds.append(_y_hat.cpu().item())
            self.gts.append(_y.cpu().item())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_precision", prec, on_epoch=True, prog_bar=True)
        self.log("val_recall", rc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

        # return loss
        return {'val_loss':loss, 'val_acc':acc,
            'val_precision':prec,'val_recall':rc, 
            'val_f1': f1}

    def validation_epoch_end(self, outputs): # epoch 마다 실행
        f_loss = 0
        f_acc = 0
        f_prec = 0
        f_rc = 0
        f_f1 = 0
        cnt = 0

        for output in outputs:
            f_loss += output['val_loss'].cpu().data.numpy()
            f_acc += output['val_acc'].cpu().data.numpy()
            f_prec += output['val_precision'].cpu().data.numpy()
            f_rc += output['val_recall'].cpu().data.numpy()
            f_f1 += output['val_f1'].cpu().data.numpy()
            cnt += 1
        
        print('[Validation Results] Loss : {:.4f}, Acc : {:.4f}, Prec : {:.4f}, \
            Recall : {:.4f}, F1 : {:.4f}'.format(
            f_loss/cnt, f_acc/cnt, f_prec/cnt, f_rc/cnt, f_f1/cnt
        ))

        self.print_pycm()


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        c_hat = to_categorical(y_hat)
        acc = self.accuracy(c_hat, y)
        prec = self.prec(c_hat, y)
        rc = self.rc(c_hat, y)
        f1 = self.f1(c_hat, y)

        for _y, _y_hat in zip(y, c_hat):
            self.preds.append(_y_hat.cpu().item())
            self.gts.append(_y.cpu().item())

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_precision", prec, on_epoch=True, prog_bar=True)
        self.log("test_recall", rc, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)

        # return {'test_loss': loss}
        return {'test_loss':loss, 'test_acc':acc,
            'test_precision':prec,'test_recall':rc, 
            'test_f1': f1}

    def test_epoch_end(self, outputs):
        f_loss = 0
        f_acc = 0
        f_prec = 0
        f_rc = 0
        f_f1 = 0
        cnt = 0

        for output in outputs:
            f_loss += output['test_loss'].cpu().data.numpy()
            f_acc += output['test_acc'].cpu().data.numpy()
            f_prec += output['test_precision'].cpu().data.numpy()
            f_rc += output['test_recall'].cpu().data.numpy()
            f_f1 += output['test_f1'].cpu().data.numpy()
            cnt += 1
        
        print('[Test Results] Loss : {:.4f}, Acc : {:.4f}, Prec : {:.4f}, \
            Recall : {:.4f},  F1 : {:.4f}'.format(
            f_loss/cnt, f_acc/cnt, f_prec/cnt, f_rc/cnt,f_f1/cnt
        ))

        self.print_pycm()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr) # hyper parameterized
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        print('\n\n')
        print(optimizer)
        print('\n\n')

        return [optimizer], [scheduler]

    def print_pycm(self):
        cm = ConfusionMatrix(self.gts, self.preds)

        for cls_name in cm.classes:
            print('============' * 5)
            print('Class Name : [{}]'.format(cls_name)) # Class name 에 대한걸 positive라고 생각하고 tp, fn, fp, tn 구하기
            TP = cm.TP[cls_name]
            TN = cm.TN[cls_name]
            FP = cm.FP[cls_name]
            FN = cm.FN[cls_name]
            acc = cm.ACC[cls_name]
            pre = cm.PPV[cls_name]
            rec = cm.TPR[cls_name]
            spec = cm.TNR[cls_name]

            if acc is 'None':
                acc = 0.0
            if pre is 'None':
                pre = 0.0
            if rec is 'None':
                rec = 0.0
            if spec is 'None':
                spec = 0.0

            print('TP : {}, FN : {}, FP : {}, TN : {}'.format(TP, FN, FP, TN))
            print('Accuracy : {:.4f}, Precision : {:.4f}, Recall(Sensitivity) : {:.4f}, Specificity : {:.4f}'.
                format(acc, pre, rec, spec))
            print('============' * 5)
        cm.print_matrix()
        auc_list = list(cm.AUC.values())
        print('AUROC : ', auc_list)
        
        auroc_mean = 0
        for auc in auc_list:
            if auc is 'None':
                auroc_mean += 0
            else:
                auroc_mean += auc
        auroc_mean = auroc_mean / len(auc_list)
        print("AUROC mean: {:.4f}".format(auroc_mean))
        
        self.gts = []
        self.preds = []