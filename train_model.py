"""
Define backborn model for train.
"""

import torch
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.metrics.utils import to_categorical
from pytorch_lightning.metrics import Accuracy, Precision, Recall, ConfusionMatrix, F1
from pycm import *
from efficientnet_pytorch import EfficientNet # 21.06.09 HG 수정 - Add Support Model [EfficientNet Family] - https://github.com/lukemelas/EfficientNet-PyTorch


from torchsummary import summary

### hparamds
'''
{
    'lr' : 1e-4,

}
'''

class CAMIO(pl.LightningModule):
    """ Define backborn model. """

    def __init__(self, config:dict):
        super(CAMIO, self).__init__()

        # self.hparams = config # config
        self.hparams.update(config) # 21.05.30 JH 수정 - self.hparams=config has been removed from later versions and is no longer supported. 
        self.save_hyperparameters() # save with hparams

        # hyper param setting
        self.init_lr = self.hparams.optimizer_lr # config['optimizer_lr']
        self.backborn = self.hparams.backborn_model # config['backborn_model']

        print(config)
        print(self.init_lr)
        print(self.backborn) # 21.06.03 HG Comment - omg, mistake name of variable, it's backbone X-)

        # model setting
        # model // choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d', 'mobilenet_v2', 'mobilenet_v3_small', 'squeezenet1_0', 'squeezenet1_1', efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7]
        if (self.backborn.find('vgg') != -1) : # # 21.06.03 HG 수정 - Add Support Model [VGG Family]
            if self.backborn == 'vgg11' : 
                print('MODEL = VGG11')
                self.model = models.vgg11(pretrained=True)
            
            elif self.backborn == 'vgg13' : 
                print('MODEL = VGG13')
                self.model = models.vgg13(pretrained=True)
            
            elif self.backborn == 'vgg16' : 
                print('MODEL = VGG16')
                self.model = models.vgg16(pretrained=True)
            
            elif self.backborn == 'vgg19' : 
                print('MODEL = VGG19')
                self.model = models.vgg19(pretrained=True)
            
            elif self.backborn == 'vgg11_bn' : 
                print('MODEL = VGG11_BN')
                self.model = models.vgg11_bn(pretrained=True)
            
            elif self.backborn == 'vgg13_bn' : 
                print('MODEL = VGG13_BN')
                self.model = models.vgg13_bn(pretrained=True)
            
            elif self.backborn == 'vgg16_bn' : 
                print('MODEL = VGG16_BN')
                self.model = models.vgg16_bn(pretrained=True)
            
            elif self.backborn == 'vgg19_bn' : 
                print('MODEL = VGG19_BN')
                self.model = models.vgg19_bn(pretrained=True)
                
            else :
                assert(False, '=== Not supported VGG model ===')

            # change to binary classification
            self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, 2)

        elif (self.backborn.find('resnet') != -1) or (self.backborn.find('resnext') != -1) :
            if self.backborn == 'resnet18' :
                print('MODEL = RESNET18')
                self.model = models.resnet18(pretrained=True)

            elif self.backborn == 'resnet34' :
                print('MODEL = RESNET34')
                self.model = models.resnet34(pretrained=True)
                
            elif self.backborn == 'resnet50' :
                print('MODEL = RESNET50')
                self.model = models.resnet50(pretrained=True)
                
            elif self.backborn == 'wide_resnet50_2':
                print('MODEL = WIDE_RESNET50_2')
                self.model = models.wide_resnet50_2(pretrained=True)

            elif self.backborn == 'resnext50_32x4d':
                print('MODEL = RESNEXT50_32x4D')
                self.model = models.resnext50_32x4d(pretrained=True)
                
            else : 
                assert(False, '=== Not supported Resnet model ===')
            
            # change to binary classification
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        elif self.backborn.find('mobilenet') != -1 : # 21.06.08 HG - 수정 - From JH code
            if self.backborn == 'mobilenet_v2' :
                print('MODEL = MOBILENET_V2')
                self.model = models.mobilenet_v2(pretrained=True)
                self.num_ftrs = self.model.classifier[-1].in_features
                self.classifier = nn.Sequential(
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(self.num_ftrs, 2),
                )
            
            elif self.backborn == 'mobilenet_v3_small' :
                print('MODEL = MOBILENET_V3_SMALL')
                # self.model = models.mobilenet_v3_small(pretrained=True)
                self.model = models.mobilenet_v3_small(pretrained=False) # model scretch learning
                self.num_ftrs = self.model.classifier[-1].in_features

                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(576, self.num_ftrs), #lastconv_output_channels, last_channel
                    torch.nn.Hardswish(inplace=True),
                    torch.nn.Dropout(p=0.2, inplace=True),
                    torch.nn.Linear(self.num_ftrs, 2) #last_channel, num_classes
                )

            elif self.backborn == 'mobilenet_v3_large' :
                print('MODEL = MOBILENET_V3_LARGE')
                self.model = models.mobilenet_v3_large(pretrained=True)
                self.num_ftrs = self.model.classifier[-1].in_features

                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(960, self.num_ftrs), #lastconv_output_channels, last_channel
                    torch.nn.Hardswish(inplace=True),
                    torch.nn.Dropout(p=0.2, inplace=True),
                    torch.nn.Linear(self.num_ftrs, 2) #last_channel, num_classes
                )

            else :
                assert(False, '=== Not supported MobileNet model ===')
        
        elif self.backborn.find('squeezenet') != -1 :# 21.06.05 HG 수정 - Squeezenet 모델에 맞게 수정 
            if self.backborn == 'squeezenet1_0' :
                print('MODEL = squeezenet1_0')
                self.model = models.squeezenet1_0(pretrained=True)

            elif self.backborn == 'squeezenet1_1' : # 21.06.05 HG 추가 - Squeezenet1_1
                print('MODEL = squeezenet1_1')
                self.model = models.squeezenet1_1(pretrained=True)

            else :
                assert(False, '=== Not supported Squeezenet model ===')
                
            # change to binary classification
            final_conv = torch.nn.Conv2d(512, 2, 1)
            self.model.classifier[1] = final_conv # change only final conv layer

        elif self.backborn.find('efficientnet') != -1 :# 21.06.09 HG 추가 - Add supported model [Efficient Family]
            if self.backborn == 'efficientnet_b0' :
                print('MODEL = EFFICIENTNET-B0')
                self.model = EfficientNet.from_pretrained('efficientnet-b0', advprop=False, num_classes=2) # Normailize from ImageNet, # change to binary classification

            elif self.backborn == 'efficientnet_b1' : 
                print('MODEL = EFFICIENTNET-B1')
                self.model = EfficientNet.from_pretrained('efficientnet-b1', advprop=False, num_classes=2) # Normailize from ImageNet

            elif self.backborn == 'efficientnet_b2' :
                print('MODEL = EFFICIENTNET-B2')
                self.model = EfficientNet.from_pretrained('efficientnet-b2', advprop=False, num_classes=2) # Normailize from ImageNet

            elif self.backborn == 'efficientnet_b3' : 
                print('MODEL = EFFICIENTNET-B3')
                self.model = EfficientNet.from_pretrained('efficientnet-b3', advprop=False, num_classes=2) # Normailize from ImageNet
            
            elif self.backborn == 'efficientnet_b4' :
                print('MODEL = EFFICIENTNET-B4')
                self.model = EfficientNet.from_pretrained('efficientnet-b4', advprop=False, num_classes=2) # Normailize from ImageNet

            elif self.backborn == 'efficientnet_b5' : 
                print('MODEL = EFFICIENTNET-B5')
                self.model = EfficientNet.from_pretrained('efficientnet-b5', advprop=False, num_classes=2) # Normailize from ImageNet

            elif self.backborn == 'efficientnet_b6' : 
                print('MODEL = EFFICIENTNET-B6')
                self.model = EfficientNet.from_pretrained('efficientnet-b6', advprop=False, num_classes=2) # Normailize from ImageNet

            elif self.backborn == 'efficientnet_b7' : 
                print('MODEL = EFFICIENTNET-B7')
                self.model = EfficientNet.from_pretrained('efficientnet-b7', advprop=False, num_classes=2) # Normailize from ImageNet

            else :
                assert(False, '=== Not supported EfficientNet model ===')            

        else :
            assert(False, '=== Not supported Model === ')


        self.criterion = torch.nn.CrossEntropyLoss()
        # self.softmax = torch.nn.Softmax()
        
        self.accuracy = Accuracy()
        self.prec = Precision(num_classes=1, multiclass=False) # 21.05.30 JH 변경 is_multiclass -> multiclass
        self.rc = Recall(num_classes=1, multiclass=False) # 21.05.30 JH 변경 is_multiclass -> multiclass
        self.f1 = F1(num_classes=1, multiclass=False) # 21.05.30 JH 변경 multilabel -> multiclass, Deprecated since version 0.3: Argument will not have any effect and will be removed in v0.4, please use multiclass intead.
        # self.confmat = ConfusionMatrix(num_classes=1)

        self.preds = []
        self.gts = []
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx): # 배치마다 실행
        x, y = batch
        y_hat = self.forward(x)
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

        # calc OOB metric 
        TP, TN, FP, FN, OOB_metric, Over_estimation, Under_estimation, Correspondence_estimation, UNCorrespondence_estimation = self.calc_OOB_metric()
        self.log("val_TP", TP, on_epoch=True, prog_bar=True)
        self.log("val_TN", TN, on_epoch=True, prog_bar=True)
        self.log("val_FP", FP, on_epoch=True, prog_bar=True)
        self.log("val_FN", FN, on_epoch=True, prog_bar=True)
        self.log("Confidence_ratio", OOB_metric, on_epoch=True, prog_bar=True)
        self.log("Over_estimation_ratio", Over_estimation, on_epoch=True, prog_bar=True)
        self.log("Under_estimation_ratio", Under_estimation, on_epoch=True, prog_bar=True)
        self.log("Correspondence", Correspondence_estimation, on_epoch=True, prog_bar=True)
        self.log("UNCorrespondence", UNCorrespondence_estimation, on_epoch=True, prog_bar=True)

        # print info, and initializae self.gts, preds
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

    def calc_OOB_metric(self) :
        IB_CLASS, OOB_CLASS = (0,1)
        OOB_metric = -1 
        
        cm = ConfusionMatrix(self.gts, self.preds)
        
        TP = cm.TP[OOB_CLASS]
        TN = cm.TN[OOB_CLASS]
        FP = cm.FP[OOB_CLASS]
        FN = cm.FN[OOB_CLASS]

        try : # zero division except       
            OOB_metric = (TP-FP) / (FN + TP + FP) # 잘못예측한 OOB / predict OOB + 실제 OOB
            Over_estimation = FP / (FN + TP + FP)
            Under_estimation = FN / (FN + TP + FP)
            Correspondence_estimation = TP / (FN + TP + FP)
            UNCorrespondence_estimation = (FP + FN) / (FN + TP + FP)
        except : 
            OOB_metric = -1
            Over_estimation = -1
            Under_estimation = -1
            Correspondence_estimation = -1
            UNCorrespondence_estimation = -1
        
        print('\n')
        print('===> \tOOB METRIC \t <===')
        print(OOB_metric)
        print('\n')

        return (TP, TN, FP, FN, OOB_metric, Over_estimation, Under_estimation, Correspondence_estimation, UNCorrespondence_estimation)


        