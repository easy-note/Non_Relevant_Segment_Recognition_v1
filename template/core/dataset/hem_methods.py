import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import numpy as np
import pandas as pd

from tqdm import tqdm

class HEMHelper():
    """
        Help computation ids for Hard Example Mining.
        
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.NON_HEM, self.HEM = (0, 1)
        self.IB_CLASS, self.OOB_CLASS = (0, 1)
        self.bsz = self.args.batch_size
        self.cnt = 0
        
    def set_method(self, method):
        if method in ['hem-softmax-offline', 'hem-voting-offline', 'hem-vi-offline']:
            self.method = 'hem-vi'
        else:
            self.method = method

    def set_ratio(self, hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df):
        # df 정렬
        hard_neg_df = hard_neg_df.sort_values(by='Img_path')
        hard_pos_df = hard_pos_df.sort_values(by='Img_path')
        vanila_neg_df = vanila_neg_df.sort_values(by='Img_path')
        vanila_pos_df = vanila_pos_df.sort_values(by='Img_path')

        # HEM 표기
        hard_neg_df['HEM'] = [1]*len(hard_neg_df)
        hard_pos_df['HEM'] = [1]*len(hard_pos_df)
        vanila_neg_df['HEM'] = [0]*len(vanila_neg_df)
        vanila_pos_df['HEM'] = [0]*len(vanila_pos_df)

        try:
            vanila_pos_df = vanila_pos_df.sample(n=len(hard_pos_df), replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, hem_oob 개
        except:
            vanila_pos_df = vanila_pos_df.sample(frac=1, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, 전체 oob_df

        target_vanila_neg_df_len = (len(hard_pos_df)+len(vanila_pos_df))*self.args.IB_ratio - len(hard_neg_df)
        
        print('target_vanila_neg_df_len : {}\n'.format(target_vanila_neg_df_len))

        
        try:
            vanila_neg_df = vanila_neg_df.sample(n=target_vanila_neg_df_len, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, target_ib_assets_df_len 개
        except:
            if target_vanila_neg_df_len <= 0: 
                vanila_neg_df = vanila_neg_df.sample(frac=0, replace=False, random_state=self.args.random_seed)
                hard_neg_df = hard_neg_df.sample(n=(len(hard_pos_df)+len(vanila_pos_df))*self.args.IB_ratio, replace=False, random_state=self.args.random_seed)
            else:
                vanila_neg_df = vanila_neg_df.sample(frac=1, replace=False, random_state=10)

        print('\n\n', '========='* 10)
        print('\thard_neg_df {}, hard_pos_df {}, vanila_neg_df {}, vanila_pos_df {}'.format(len(hard_neg_df), len(hard_pos_df), len(vanila_neg_df), len(vanila_pos_df)))
        print('========='* 10, '\n\n')

        hem_dataset_len_list = [len(hard_neg_df), len(hard_pos_df), len(vanila_neg_df), len(vanila_pos_df)]

        final_pos_assets_df = pd.concat([hard_pos_df, vanila_pos_df])[['Img_path', 'GT', 'HEM']]
        final_neg_assets_df = pd.concat([hard_neg_df, vanila_neg_df])[['Img_path', 'GT', 'HEM']]

        # sort & shuffle
        final_assets_df = pd.concat([final_pos_assets_df, final_neg_assets_df]).sort_values(by='Img_path', axis=0, ignore_index=True)
        print('\tSORT final_assets HEAD\n', final_assets_df.head(20), '\n\n')

        final_assets_df = final_assets_df.sample(frac=1, random_state=self.args.random_seed).reset_index(drop=True)
        print('\tSHUFFLE final_assets HEAD\n', final_assets_df.head(20), '\n\n')
        
        final_assets_df.columns = ['img_path', 'class_idx', 'HEM']



        return final_assets_df, hem_dataset_len_list
        
    def compute_hem(self, *args):
        if self.method == 'hem-vi':
            return self.hem_vi(*args)
        elif self.method == 'hem-emb-online':
            return self.hem_cos_hard_sim(*args)
        else: # exception
            return None

    def hem_vi(self, model, dataset): # MC dropout
        print('hem_mc methods')

        # Function to enable the dropout layers during test-time
        def enable_dropout(model):
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        
        # extract hem idx method
        def extract_hem_idx_from_voting(dropout_predictions, gt_list, img_path_list):
            hem_idx = []
            
            # 1. extract hem index
            predict_table = np.argmax(dropout_predictions, axis=2) # (forward_passes, n_samples)
            predict_ratio = np.mean(predict_table, axis=0) # (n_samples)

            predict_np = np.around(predict_ratio) # threshold == 0.5, if predict_ratio >= 0.5, predict_class == OOB(1)
            predict_np = np.int8(predict_np) # casting float to int
            predict_list = predict_np.tolist() # to list

            answer = predict_np == np.array(gt_list) # compare with gt list

            hem_idx = np.where(answer == False) # hard example
            hem_idx = hem_idx[0].tolist() # remove return turple

            # 2. split hem/vanila 
            total_df_dict = {
                'Img_path': img_path_list,
                'predict': predict_list,
                'GT': gt_list,
                'voting': predict_ratio.tolist(),
                'hem': [self.NON_HEM] * len(img_path_list) # init hem
            }

            total_df = pd.DataFrame(total_df_dict)
            total_df.loc[hem_idx, ['hem']] = self.HEM # hem index

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = split_to_hem_vanila_df(total_df)

            return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df
        
        def extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list):
            hem_idx = []
            
            cols = ['Img_path', 'GT', 'Predict', 'Logit', 'Diff', 'Consensus']
            CORRECT, INCORRECT = (0,1)
            IB_CLASS, OOB_CLASS = (0,1)

            dropout_predictions_mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
            dropout_predictions_mean_argmax = np.argmax(dropout_predictions_mean, axis=1) # shape (n_samples,)
            dropout_predictions_mean_abs_diff = np.squeeze(np.abs(np.diff(dropout_predictions_mean, axis=1)))# shape (n_samples,)

            logit_list = dropout_predictions_mean.tolist()
            predict_list = dropout_predictions_mean_argmax.tolist()
            diff_list = dropout_predictions_mean_abs_diff.tolist()
            consensus_list = [CORRECT if y==y_hat else INCORRECT for y, y_hat in zip(gt_list, predict_list)]

            vanila_df = pd.DataFrame([x for x in zip(img_path_list, gt_list, predict_list, logit_list, diff_list, consensus_list)],
                    columns=cols)

            top_ratio = self.args.top_ratio # 30/100
            top_k = int(len(vanila_df) * top_ratio)

            hard_df_lower_idx = dropout_predictions_mean_abs_diff.argsort()[:top_k].tolist()
            hard_df_upper_idx = (-dropout_predictions_mean_abs_diff).argsort()[:top_k].tolist()

            hard_idx = []
            for i, gt in enumerate(gt_list):
                if (i in hard_df_upper_idx) and gt==INCORRECT:
                    hard_idx.append(i)

            hard_idx = hard_idx + hard_df_lower_idx

            vanila_idx = []
            for i in range(len(vanila_df)):
                if (i not in hard_idx):
                    vanila_idx.append(i)

            hard_df = vanila_df.loc[hard_idx, :]
            vanila_df = vanila_df.loc[vanila_idx, :]

            hard_neg_df = hard_df[hard_df['GT']==IB_CLASS]
            hard_pos_df = hard_df[hard_df['GT']==OOB_CLASS]

            vanila_neg_df = vanila_df[vanila_df['GT']==IB_CLASS]
            vanila_pos_df = vanila_df[vanila_df['GT']==OOB_CLASS]

            

            return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df


        def extract_hem_idx_from_mutual_info(dropout_predictions, gt_list, img_path_list):
            hem_idx = []

            # 1. extract hem index

            ## Calculate maen and variance
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
            # so in (hard example data selection?), if i'th data is high(relavence), non-indepandence, i'th data has similar so it's hard

            # sort mi index & extract top/btm sample index 
            top_ratio = 30/100
            top_k = int(len(mutual_info) * top_ratio)

            btm_ratio = 30/100
            btm_k = int(len(mutual_info) * btm_ratio)

            sorted_mi_index = (-mutual_info).argsort() # desecnding index
            top_mi_index = sorted_mi_index[:top_k] # highest 
            btm_mi_index = sorted_mi_index[len(mutual_info) - btm_k:] # lowest

            # extract wrong anwer from mean softmax 
            predict_np = np.argmax(mean, axis=1)
            predict_list = predict_np.tolist() # to list

            answer = predict_np == np.array(gt_list) # compare with gt list

            wrong_idx = np.where(answer == False) # wrong example
            wrong_idx = wrong_idx[0].tolist() # remove return turple

            # append hem idx - high mi & wrong answer
            hem_idx += np.intersect1d(wrong_idx, top_mi_index).tolist()
            # append hem idx - low mi
            hem_idx += btm_mi_index.tolist()
            print('hem_idx')
            
            # 2. split hem/vanila 
            total_df_dict = {
                'Img_path': img_path_list,
                'predict': predict_list,
                'GT': gt_list,
                'mi': mutual_info.tolist(),
                'hem': [self.NON_HEM] * len(img_path_list) # init hem
            }

            total_df = pd.DataFrame(total_df_dict)
            total_df.loc[hem_idx, ['hem']] = self.HEM # hem index

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = split_to_hem_vanila_df(total_df)

            return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df

        def split_to_hem_vanila_df(total_df): # total_df should have ['hem', 'GT'] columne
            hem_df = total_df[total_df['hem'] == self.HEM]
            vanila_df = total_df[total_df['hem'] == self.NON_HEM]

            hard_neg_df = hem_df[hem_df['GT'] == self.IB_CLASS]
            hard_pos_df = hem_df[hem_df['GT'] == self.OOB_CLASS]
            
            vanila_neg_df = vanila_df[vanila_df['GT'] == self.IB_CLASS]
            vanila_pos_df = vanila_df[vanila_df['GT'] == self.OOB_CLASS]

            return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df
        
        # init for parameter for hem methods
        img_path_list = []
        gt_list = []

        for data in dataset:
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
            for data in dataset:
                with torch.no_grad():
                    y_hat = model(data['x'].cuda())
                    y_hat = softmax(y_hat)

                predictions = np.vstack((predictions, y_hat.cpu().numpy()))

            # dropout predictions - shape (forward_passes, n_samples, n_classes)
            dropout_predictions = np.vstack((dropout_predictions,
                                        predictions[np.newaxis, :, :]))
        

        # extracting hem, apply hem extract mode
        if self.args.hem_extract_mode == 'hem-softmax-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list)

        elif self.args.hem_extract_mode == 'hem-voting-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = extract_hem_idx_from_voting(dropout_predictions, gt_list, img_path_list)
        
        elif self.args.hem_extract_mode == 'hem-vi-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = extract_hem_idx_from_voting(dropout_predictions, gt_list, img_path_list)

        print('hard_neg_df', len(hard_neg_df))
        print('hard_pos_df', len(hard_pos_df))
        print('vanila_neg_df', len(vanila_neg_df))
        print('vanila_pos_df', len(vanila_pos_df))

        total_dataset_len_list = [len(hard_neg_df), len(hard_pos_df), len(vanila_neg_df), len(vanila_pos_df)]

        hem_final_df, hem_dataset_len_list = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

        return hem_final_df, total_dataset_len_list, hem_dataset_len_list
    
    def hem_cos_hard_sim(self, model, x, y, loss_fn):
        emb, y_hat = model(x)
        sim_dist = emb @ model.proxies
        sim_preds = torch.argmax(sim_dist, -1)
        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
        
        pos_y_hat = y_hat[correct_answer]
        pos_y = y[correct_answer]
        
        pos_loss = loss_fn(pos_y_hat, pos_y)
        
        neg_y_hat = y_hat[wrong_answer]
        neg_y = y[wrong_answer]
        
        wrong_sim_dist = sim_dist[wrong_answer, neg_y]
        wrong_ids = torch.argsort(wrong_sim_dist)
        
        w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
        neg_y_hat = neg_y_hat[wrong_ids]
        neg_y = neg_y[wrong_ids]
        
        neg_loss = 0
        for wi in range(len(w)):
            neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w[wi:wi+1]
            
        return (pos_loss + neg_loss) / 2. + loss_fn(sim_dist[correct_answer], pos_y)
        # return (pos_loss + neg_loss) / 2. + loss_fn(sim_dist, y)
    
