import torch
import math
import numpy as np
from pycm import *
from core.utils.misc import *

import matplotlib.pyplot as plt

class MetricHelper():
    """
        Help metric computation.
    """
    def __init__(self):
        # TODO 필요한거 더 넣기
        self.EXCEPTION_NUM = -100
        self.IB_CLASS, self.OOB_CLASS = (0, 1)
        self.pred_list = []
        self.gt_list = []

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.epoch = 0

    def write_preds(self, pred_list, gt_list):
        for pred, gt in zip(pred_list, gt_list):
            self.pred_list.append(pred.item())
            self.gt_list.append(gt.item())

    def write_loss(self, loss_val, task='train'):
        if task == 'train':
            self.train_loss.append(loss_val)
        elif task == 'val':
            self.valid_loss.append(loss_val)
            self.epoch += 1
        elif task == 'test':
            self.test_loss.append(loss_val)

    def calc_metric(self):
        classes = [self.IB_CLASS, self.OOB_CLASS]

        cm = ConfusionMatrix(self.gt_list, self.pred_list, classes=classes) # pycm
        
        try: # for [1,1,1] [1,1,1] error solving
            cm.relabel(mapping={0:self.IB_CLASS, 1:self.OOB_CLASS}) # when [1,1,0] [0,1,0] return => cm.classes : [0, 1]
        except:
            cm.relabel(mapping={'0':self.IB_CLASS, '1':self.OOB_CLASS}) # when [1,1,1] [1,1,1] return => cm.classes : ['0', '1']

        metrics = {
            'TP': cm.TP[self.OOB_CLASS],
            'TN': cm.TN[self.OOB_CLASS],
            'FP': cm.FP[self.OOB_CLASS],
            'FN': cm.FN[self.OOB_CLASS],
            'Accuracy': cm.ACC[self.OOB_CLASS],
            'Precision': cm.PPV[self.OOB_CLASS],
            'Recall': cm.TPR[self.OOB_CLASS],
            'F1-Score': cm.F1[self.OOB_CLASS],
        }

        # np casting for zero divide to inf
        TP = np.float16(metrics['TP'])
        FP = np.float16(metrics['FP'])
        TN = np.float16(metrics['TN'])
        FN = np.float16(metrics['FN'])

        metrics['OOB_metric'] = (TP - FP) / (FN + TP + FP) # 잘못예측한 OOB / predict OOB + 실제 OOB
        metrics['Over_estimation'] = FP / (FN + TP + FP) # OR
        metrics['Under_estimation'] = FN / (FN + TP + FP)
        metrics['Correspondence_estimation'] = TP / (FN + TP + FP) # CR
        metrics['UNCorrespondence_estimation'] = (FP + FN) / (FN + TP + FP)

        # exception
        for k, v in metrics.items():
            if v == 'None': # ConfusionMetrix return
                metrics[k] = self.EXCEPTION_NUM
            elif np.isinf(v): # numpy return
                metrics[k] = self.EXCEPTION_NUM

        self.pred_list = []
        self.gt_list = []

        return metrics

    def aggregate_calc_metric(self, metrics_list):
        advanced_metrics = {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0,
            'Accuracy': self.EXCEPTION_NUM,
            'Precision': self.EXCEPTION_NUM,
            'Recall': self.EXCEPTION_NUM,
            'F1-Score': self.EXCEPTION_NUM,
        }

        # sum of TP/TN/FP/FN
        for metrics in metrics_list:
            advanced_metrics['TP'] += metrics['TP']
            advanced_metrics['TN'] += metrics['TN']
            advanced_metrics['FP'] += metrics['FP']
            advanced_metrics['FN'] += metrics['FN']

        # np casting for zero divide to inf
        TP = np.float16(advanced_metrics['TP'])
        FP = np.float16(advanced_metrics['FP'])
        TN = np.float16(advanced_metrics['TN'])
        FN = np.float16(advanced_metrics['FN'])

        advanced_metrics['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
        advanced_metrics['Precision'] = TP / (TP + FP)
        advanced_metrics['Recall'] = TP / (TP + FN)
        advanced_metrics['F1-Score'] = 2 * ((advanced_metrics['Precision'] * advanced_metrics['Recall']) / (advanced_metrics['Precision'] + advanced_metrics['Recall']))

        advanced_metrics['OOB_metric'] = (TP - FP) / (FN + TP +FP)
        advanced_metrics['Over_estimation'] = FP / (FN + TP + FP)
        advanced_metrics['Under_estimation'] = FN / (FN + TP + FP)
        advanced_metrics['Correspondence_estimation'] = TP / (FN + TP + FP)
        advanced_metrics['UNCorrespondence_estimation'] = (FP + FN) / (FN + TP + FP)

        # calc mCR, mOR
        advanced_metrics['mCR'] = np.mean([metrics['OOB_metric'] for metrics in metrics_list])
        advanced_metrics['mOR'] = np.mean([metrics['Over_estimation'] for metrics in metrics_list])

        # exception
        for k, v in advanced_metrics.items():
            if v == 'None': # ConfusionMetrix return
                advanced_metrics[k] = self.EXCEPTION_NUM
            elif np.isinf(v): # numpy return
                advanced_metrics[k] = self.EXCEPTION_NUM

        return advanced_metrics

    def save_metric(self, metric, epoch, args, save_path, task='OOB'):
        if task=='OOB':
            save_OOB_result_csv(metric=metric, epoch=epoch, args=args, save_path=save_path)

    def save_loss_pic(self, save_path):
        fig = plt.figure(figsize=(32, 16))

        plt.ylabel('Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)

        plt.plot(range(self.epoch), self.train_loss)
        plt.plot(range(self.epoch), self.valid_loss)
        
        plt.legend(['Train', 'Val'], fontsize=20)
        plt.savefig(save_path + '/loss.png')
