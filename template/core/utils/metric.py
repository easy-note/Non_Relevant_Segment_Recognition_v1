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
        self.OOB_CLASS = 1
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
        cm = ConfusionMatrix(self.gt_list, self.pred_list)

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
        
        metrics['OOB_metric'] = (metrics['TP']-metrics['FP']) / (metrics['FN'] + metrics['TP'] + metrics['FP']) # 잘못예측한 OOB / predict OOB + 실제 OOB
        metrics['Over_estimation'] = metrics['FP'] / (metrics['FN'] + metrics['TP'] + metrics['FP']) # OR
        metrics['Under_estimation'] = metrics['FN'] / (metrics['FN'] + metrics['TP'] + metrics['FP'])
        metrics['Correspondence_estimation'] = metrics['TP'] / (metrics['FN'] + metrics['TP'] + metrics['FP']) # CR
        metrics['UNCorrespondence_estimation'] = (metrics['FP'] + metrics['FN']) / (metrics['FN'] + metrics['TP'] + metrics['FP'])
        metrics['Mean_metric'] = (metrics['Correspondence_estimation'] + (1-metrics['Over_estimation'])) / 2.

        # exception
        for k, v in metrics.items():
            if v == 'None': # ConfusionMetrix return
                metrics[k] = self.EXCEPTION_NUM
            elif np.isinf(v): # numpy return
                metrics[k] = self.EXCEPTION_NUM

        self.pred_list = []
        self.gt_list = []

        return metrics

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
