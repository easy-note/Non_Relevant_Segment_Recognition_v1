import torch
import math
import numpy as np
from pycm import *


class MetricHelper():
    """
        Help metric computation.
    """
    def __init__(self):
        # TODO 필요한거 더 넣기
        self.OOB_CLASS = 1
        self.pred_list = []
        self.gt_list = []

    def write_preds(self, pred_list, gt_list):
        for pred, gt in zip(pred_list, gt_list):
            self.pred_list.append(pred.item())
            self.gt_list.append(gt.item())

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
        
        metrics['OOB_metric'] = (TP-FP) / (FN + TP + FP) # 잘못예측한 OOB / predict OOB + 실제 OOB
        metrics['Over_estimation'] = FP / (FN + TP + FP)
        metrics['Under_estimation'] = FN / (FN + TP + FP)
        metrics['Correspondence_estimation'] = TP / (FN + TP + FP)
        metrics['UNCorrespondence_estimation'] = (FP + FN) / (FN + TP + FP)

        # exception
        for k, v in metrics.items():
            if np.isnan(v) or np.isinf(v):
                metrics[k] = -1

        self.pred_list = []
        self.gt_list = []

        return metrics

    # ------ 아래는 필요하면 채워서 사용, 아니면 지워도 됨 -------
    def calc_CR(self, data):        
        pass
    
    def calc_OR(self, data):
        pass

    def calc_meanCROR(self, data):
        pass

    