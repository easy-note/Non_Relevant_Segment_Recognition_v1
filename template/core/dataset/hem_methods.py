import torch
import torch.nn as nn

import numpy as np
import sys

import pandas as pd

from tqdm import tqdm

class HEMHelper():
    """
        Help computation ids for Hard Example Mining.
        
    """
    def __init__(self):
        super().__init__()
        
    def set_method(self, method):
        self.method = method

    def compute_hem(self, model, data_loader):
        if self.method == 'hem-softmax':
            return self.hem_softmax_diff(model, data_loader)
        elif self.method == 'hem-vi':
            return self.hem_vi(model, data_loader)
        elif self.method == 'hem-bs':
            return self.hem_batch_sampling(model, data_loader)
        else: # exception
            return None

    def hem_softmax_diff(self, model, data_loader):
        pass

    def hem_vi(self, model, data_loader): # MC dropout
        print('hem_mc methods')

        # Function to enable the dropout layers during test-time
        def enable_dropout(model):
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        
        # extract hem idx method
        def extract_hem_idx_from_voting(dropout_predictions):
            hem_idx = []
            
            predict_table = np.argmax(dropout_predictions, axis=2) # (forward_passes, n_samples)
            predict_ratio = np.mean(predict_table, axis=0) # (n_samples)

            predict_list = np.around(predict_ratio) # threshold == 0.5, if predict_ratio >= 0.5, predict_class == OOB(1)

            answer = predict_list == np.array(gt_list) # compare with gt list

            hem_idx = np.where(answer == False) # hard example
            hem_idx = hem_idx[0].tolist() # remove return turple

            return hem_idx
        
        def extract_hem_idx_from_softmax_diff(dropout_predictions):
            hem_idx = []

            # TO DO

            return hem_idx

        def extract_hem_idx_from_mutual_info(dropout_predictions):
            hem_idx = []

            ## 2. Calculate maen and variance
            print(dropout_predictions)
            mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes) 
            variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)
            
            epsilon = sys.float_info.min
            ## calc entropy across multiple MCD forward passes
            entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

            ## calc mutual information
            # (https://www.edwith.org/medical-20200327/lecture/63144/)
            mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                                                                        axis=-1), axis=0) # shape (n_samples,)
            # if mutual info is high = relavence // low = non-relavence
            # so in (hard example data selection?), if i'th data is high(relavence), non-indepandence, i'th data has similar so it's hard?
            print(mutual_info)

            # top(high) 30 % dataset
            # if mutual_info 
            top_ratio = 30/100
            top_k = int(len(mutual_info) * top_ratio)

            hem_idx = (-mutual_info).argsort()[:top_k].tolist() # descending

            return hem_idx, mutual_info

        def split_hard_example_class(hem_idx, gt_list):
            hard_neg_idx, hard_pos_idx = [], []
            IB_CLASS, OOB_CLASS = 0, 1

            for idx in hem_idx: 
                hem_example_class = gt_list[idx] # hard example's gt class
                if hem_example_class == IB_CLASS: # hard negative sample
                    hard_neg_idx.append(idx)
                elif hem_example_class == OOB_CLASS: # hard positive sample
                    hard_pos_idx.append(idx)

            return hard_neg_idx, hard_pos_idx

        hem_df = None


        col_name = ['img_path', 'class_idx']
        img_path_list = []
        gt_list = []
        
        # init for return df(hem df)
        for data in outputs:
            img_path_list += list(data['img_path'])
            gt_list += data['y'].tolist()
        
        ### 0. MC paramter setting
        n_classes = 2
        forward_passes = 5
        n_samples = len(img_path_list)

        
        dropout_predictions = np.empty((0, n_samples, n_classes)) 
        softmax = nn.Softmax(dim=1)

        ## 1. MC forward
        for cnt in tqdm(range(forward_passes), desc='MC FORWARDING ... '):
            predictions = np.empty((0, n_classes))
            model.eval()
            enable_dropout(model)
            for data in outputs:
                with torch.no_grad():
                    y_hat = model(data['x'])
                    y_hat = softmax(y_hat)

                predictions = np.vstack((predictions, y_hat.cpu().numpy()))

            # dropout predictions - shape (forward_passes, n_samples, n_classes)
            dropout_predictions = np.vstack((dropout_predictions,
                                        predictions[np.newaxis, :, :]))
        
        
        # extracting he // apply method
        hem_idx, mutual_info = extract_hem_idx_from_mutual_info(dropout_predictions)

        # split hard example class from he 
        hard_neg_idx, hard_pos_idx = split_hard_example_class(hem_idx, gt_list)

        print(hard_neg_idx)
        print(hard_pos_idx)

        hem_df = pd.DataFrame(
                {
                    'img_path': img_path_list,
                    'gt_list': gt_list,
                    'mutual_info': mutual_info.tolist()
                }
            )

        '''
        img_path | class
        ~.jpg | 0
        ~.jpg | 1
        # return df of Hard Example
        '''

        return hem_df
        


    def hem_batch_sampling(self, model, data_loader):
        pass