import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
import json

import os

from torchsummary import summary

import pickle

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
        if method in ['hem-softmax-offline', 'hem-voting-offline', 'hem-vi-offline', 'all-offline']:
            self.method = 'hem-vi'
        else:
            self.method = method

    def set_restore_path(self, restore_path):
        self.restore_path = restore_path

    def get_target_hem_count(self):
        with open(os.path.join(self.restore_path, 'DATASET_COUNT.json')) as file:
            try:
                json_data = json.load(file)
                return json_data['target_hem_count']['rs'], json_data['target_hem_count']['nrs']

            except ValueError as e:
                print('Parsing Fail, Error: {}'.format(e))
                return None
    
    def get_target_patient_hem_count(self, patient_no): # rule for target count of each patients
        # hem
        patient_rs_count, patient_nrs_count = 0,0
        patient_rs_ratio, patient_nrs_ratio = 0,0
        target_rs_cnt, target_nrs_cnt = self.get_target_hem_count()
        
        with open(os.path.join(self.restore_path, 'PATIENTS_DATASET_COUNT.json')) as file:
            try:
                json_data = json.load(file)
                patient_rs_ratio, patient_nrs_ratio = json_data['target_hem_count'][patient_no]['rs_ratio'], json_data['target_hem_count'][patient_no]['nrs_ratio']

            except ValueError as e:
                print('Parsing Fail, Error: {}'.format(e))
                return None

            patient_rs_count = target_nrs_cnt * patient_nrs_ratio
            patient_nrs_count = patient_rs_count * self.args.IB_ratio

        return patient_rs_count, patient_nrs_count

    def set_ratio(self, hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient_no=None):
        if patinet_no :
            target_rs_cnt, target_nrs_cnt = self.get_target_patient_hem_count(patient_no)
        else : # if patinet_no = None
            target_rs_cnt, target_nrs_cnt = self.get_target_hem_count()

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


        # train data 수 만큼 hem data extract
        if len(hard_pos_df) > target_nrs_cnt: hard_pos_df = hard_pos_df.sample(n=target_nrs_cnt, replace=False, random_state=self.args.random_seed)
        if len(hard_neg_df) > target_rs_cnt: hard_neg_df = hard_neg_df.sample(n=target_rs_cnt, replace=False, random_state=self.args.random_seed) 

        target_len_vanila_pos = target_nrs_cnt - len(hard_pos_df)
        target_len_vanila_neg = target_rs_cnt - len(hard_neg_df)

        # if target_len_vanila_pos < 0: target_len_vanila_pos = 0
        # if target_len_vanila_neg < 0: target_len_vanila_neg = 0
        
        try:
            vanila_pos_df = vanila_pos_df.sample(n=target_len_vanila_pos, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, hem_oob 개
        except:
            vanila_pos_df = vanila_pos_df.sample(frac=1, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, 전체 oob_df

        try:
            vanila_neg_df = vanila_neg_df.sample(n=target_len_vanila_neg, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, target_ib_assets_df_len 개
        except:
            vanila_neg_df = vanila_neg_df.sample(frac=1, replace=False, random_state=self.args.random_seed)

        save_data = {
            "FINAL_HEM_DATASET": {
                "rs": {
                    "hem": len(hard_neg_df),
                    "vanila": len(vanila_neg_df)
                },
                "nrs": {
                    "hem": len(hard_pos_df),
                    "vanila": len(vanila_pos_df)
                }
            }
        }

        with open(os.path.join(self.restore_path, 'DATASET_COUNT.json'), 'r+') as f:
            data = json.load(f)
            data.update(save_data)

        with open(os.path.join(self.restore_path, 'DATASET_COUNT.json'), 'w') as f:
            json.dump(data, f, indent=2)


        final_pos_assets_df = pd.concat([hard_pos_df, vanila_pos_df])[['Img_path', 'GT', 'HEM']]
        final_neg_assets_df = pd.concat([hard_neg_df, vanila_neg_df])[['Img_path', 'GT', 'HEM']]

        # sort & shuffle
        final_assets_df = pd.concat([final_pos_assets_df, final_neg_assets_df]).sort_values(by='Img_path', axis=0, ignore_index=True)
        print('\tSORT final_assets HEAD\n', final_assets_df.head(20), '\n\n')

        final_assets_df = final_assets_df.sample(frac=1, random_state=self.args.random_seed).reset_index(drop=True)
        print('\tSHUFFLE final_assets HEAD\n', final_assets_df.head(20), '\n\n')
        
        final_assets_df.columns = ['img_path', 'class_idx', 'HEM']


        return final_assets_df
        
    def compute_hem(self, *args):
        if self.method == 'hem-vi':
            return self.hem_vi(*args)
        elif self.method == 'hem-emb-online':
            if self.args.emb_type == 1 or self.args.emb_type == 2:
                return self.hem_cos_hard_sim(*args)
            elif self.args.emb_type == 3:
                return self.hem_cos_hard_sim2(*args)
        else: # exception
            return None

    def hem_vi(self, model, dataset): # MC dropout
        print('hem_mc methods')

        # Function to enable the dropout layers during test-time
        def enable_dropout(model):
            
            ### 2. summary model
            # print('\n\n==== MODEL SUMMARY ====\n\n')
            # summary(model, (3,224,224))

            dropout_layer = []
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    dropout_layer.append(m)
            
            if self.args.n_dropout != 1 :
                dropout_layer[-1].train() # only last layer to train

        # init for parameter for hem methods
        img_path_list = []
        gt_list = []

        d_loader = DataLoader( # validation
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )
        
        img_path_list = dataset.img_list
        gt_list = dataset.label_list
        
        ### 0. MC paramter setting
        n_classes = 2
        forward_passes = self.args.n_dropout
        n_samples = len(img_path_list)
        
        dropout_predictions = np.empty((0, n_samples, n_classes)) 
        softmax = nn.Softmax(dim=1)

        ## 1. MC forward
        for cnt in tqdm(range(forward_passes), desc='MC FORWARDING ... '):
            predictions = np.empty((0, n_classes))
            model.eval()
            enable_dropout(model)
            
            print('{}th MC FORWARDING ...'.format(cnt+1))
            
            for data in tqdm(d_loader, desc='processing...'):
                with torch.no_grad():
                    y_hat = model(data[1].cuda())
                    y_hat = softmax(y_hat)

                predictions = np.vstack((predictions, y_hat.cpu().numpy()))

            # dropout predictions - shape (forward_passes, n_samples, n_classes)
            dropout_predictions = np.vstack((dropout_predictions,
                                        predictions[np.newaxis, :, :]))        

        # extracting hem, apply hem extract mode
        if self.args.hem_extract_mode == 'hem-softmax-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list)

        elif self.args.hem_extract_mode == 'hem-voting-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_voting(dropout_predictions, gt_list, img_path_list)
        
        elif self.args.hem_extract_mode == 'hem-vi-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(dropout_predictions, gt_list, img_path_list)

        elif self.args.hem_extract_mode == 'all-offline':

            '''
            #### mc assets 저장 start
            mc_assets_save_dir = os.path.join(self.restore_path, 'mc_assets')
            os.makedirs(mc_assets_save_dir, exist_ok=True)
            
            dropout_save_path = os.path.join(mc_assets_save_dir, 'dropout_predictions.npy')
            gt_list_save_path = os.path.join(mc_assets_save_dir, 'gt_list.pkl')
            img_path_list_save_path = os.path.join(mc_assets_save_dir, 'img_path_list.pkl')
            
            np.save(dropout_save_path, dropout_predictions)

            with open(gt_list_save_path, 'wb') as f:
                pickle.dump(gt_list, f)

            with open(img_path_list_save_path, 'wb') as f:
                pickle.dump(img_path_list, f)            
            #### mc assets 저장 end
            '''

            '''
            def get_patient_no(img_db_path):
                cleand_file_name = os.path.splitext(os.path.basename(img_db_path))[0]
                file_info, frame_idx = cleand_file_name.split('-')
                
                hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no = file_info.split('_')
                patient_no = '_'.join([op_method, patient_idx])

                return patient_no

            ### 환자별 mc assets split
            mc_assets = {
                'img_path': img_path_list,
                'gt':gt_list,
            }

            mc_df = pd.DataFrame(mc_assets)

            mc_df['patinet_no'] = mc_df['img_path'].apply(get_patient_no) # extract patinet_no from image db path

            patients_grouped = mc_df.groupby('patinet_no') # grouping by patinets no

            # per patients
            for patient_no, patient_df in tqdm(patients_grouped, desc='Extracting Hem Assets per Patients'): # per each patient
                
                print('Patient:{} - Sampling: {}'.format(patient_no, len(patient_df)))
                
                # patient_df = patient_df.reset_index(drop=True) # (should) reset index
                patient_index = patient_df.index # becuase mc_assets 과 행 index가 동일
                patient_index = patient_index.tolist()

                print('patient_df')
                print(patient_df)

                print('patient_index')
                print(patient_index)

                patient_img_path_list = patient_df['img_path'].tolist()
                patient_gt_list = patient_df['gt'].tolist()

                print('patient_img_path_list')
                print('patient_gt_list')
                print(patient_img_path_list)
                print(patient_gt_list)

                # dropout predictions - shape (forward_passes, n_samples, n_classes)
                patient_dropout_predictions = dropout_predictions[:, patient_index, :] # indexing from mc_df

                print('patient_dropout_pred')
                print(patient_dropout_predictions)
                print(patient_dropout_predictions.shape)

                hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(patient_dropout_predictions, patient_gt_list, patient_img_path_list, 'diff_small')
                softmax_diff_small_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

                print('softmax_diff_small_hem_final_df')
                print(softmax_diff_small_hem_final_df)

                exit(0)
                '''


            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list, 'diff_small')
            softmax_diff_small_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list, 'diff_large')
            softmax_diff_large_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_voting(dropout_predictions, gt_list, img_path_list)
            voting_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(dropout_predictions, gt_list, img_path_list, 'small')   
            vi_small_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(dropout_predictions, gt_list, img_path_list, 'large')   
            vi_large_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            return softmax_diff_small_hem_final_df, softmax_diff_large_hem_final_df, voting_hem_final_df, vi_small_hem_final_df, vi_large_hem_final_df

        return hem_final_df
            
        
    # extract hem idx method
    def extract_hem_idx_from_voting(self, dropout_predictions, gt_list, img_path_list):
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

        hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.split_to_hem_vanila_df(total_df)

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df
    
    def extract_hem_idx_from_softmax_diff(self, dropout_predictions, gt_list, img_path_list, diff):
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

        # softmax hem 추출 시, lower_idx, upper_idx 따로 실험 :) 
        if diff == 'diff_small': # lower
            hard_df_lower_idx = dropout_predictions_mean_abs_diff.argsort()[:top_k].tolist() # 올림차순
            hard_idx = hard_df_lower_idx

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

        elif diff == 'diff_large': # higher
            hard_df_upper_idx = (-dropout_predictions_mean_abs_diff).argsort()[:top_k].tolist() # 내림차순

            hard_idx = []
            for i, gt in enumerate(gt_list):
                if (i in hard_df_upper_idx) and gt==INCORRECT:
                    hard_idx.append(i)

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

    def extract_hem_idx_from_mutual_info(self, dropout_predictions, gt_list, img_path_list, location):

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
        top_ratio = self.args.top_ratio
        top_k = int(len(mutual_info) * top_ratio)

        btm_ratio = self.args.top_ratio
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
        if location == 'large':
            hem_idx += np.intersect1d(wrong_idx, top_mi_index).tolist()
        
        # append hem idx - low mi
        elif location == 'small':
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

        hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.split_to_hem_vanila_df(total_df)

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df

    def split_to_hem_vanila_df(self, total_df): # total_df should have ['hem', 'GT'] columne
        hem_df = total_df[total_df['hem'] == self.HEM]
        vanila_df = total_df[total_df['hem'] == self.NON_HEM]

        hard_neg_df = hem_df[hem_df['GT'] == self.IB_CLASS]
        hard_pos_df = hem_df[hem_df['GT'] == self.OOB_CLASS]
        
        vanila_neg_df = vanila_df[vanila_df['GT'] == self.IB_CLASS]
        vanila_pos_df = vanila_df[vanila_df['GT'] == self.OOB_CLASS]

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df
        

    
    def hem_cos_hard_sim(self, model, x, y, loss_fn):
        emb, y_hat = model(x)
        # emb = B x ch
        # proxies = ch x classes
        
        if self.args.emb_type == 1:
            sim_dist = emb @ model.proxies
            sim_preds = torch.argmax(sim_dist, -1)
        elif self.args.emb_type == 2:
            sim_dist = torch.zeros((emb.size(0), model.proxies.size(1))).to(emb.device)
            
            for d in range(sim_dist.size(1)):
                sim_dist[:, d] = 1 - torch.nn.functional.cosine_similarity(emb, model.proxies[:, d].unsqueeze(0))
            
            sim_preds = torch.argmin(sim_dist, -1)
        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
        
        if sum(correct_answer) > 0:
            pos_y_hat = y_hat[correct_answer]
            pos_y = y[correct_answer]
            
            pos_loss = loss_fn(pos_y_hat, pos_y)
            
            if self.args.use_proxy_all:
                proxy_loss = loss_fn(sim_dist, y)
            else:
                proxy_loss = loss_fn(sim_dist[correct_answer], pos_y)
        else:
            pos_loss = 0
            proxy_loss = 0
        
        if sum(wrong_answer) > 0:
            neg_y_hat = y_hat[wrong_answer]
            neg_y = y[wrong_answer]
            
            wrong_sim_dist = sim_dist[wrong_answer, neg_y]
            wrong_ids = torch.argsort(wrong_sim_dist)
            
            w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
            neg_y_hat = neg_y_hat[wrong_ids]
            neg_y = neg_y[wrong_ids]
            
            neg_loss = 0
            neg_sim_dist = sim_dist[wrong_ids]
            
            if self.args.use_half_neg:
                if len(w) > 1:
                    st = len(w) // 2
                for wi in range(st, len(w)):
                    neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w[wi:wi+1]
                    
                    if self.args.use_neg_proxy:
                        n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
            else:
                for wi in range(len(w)):
                    neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w[wi:wi+1]
                    
                    if self.args.use_neg_proxy:
                        n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
              
        else:
            neg_loss = 0
            
        return (pos_loss + neg_loss) / 2. + proxy_loss

    
    def hem_cos_hard_sim2(self, model, x, y, loss_fn):
        emb, y_hat = model(x)
        sim_dist = emb @ model.proxies
        sim_preds = torch.argmax(sim_dist, -1)
        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
        
        pos_y_hat = y_hat[correct_answer]
        pos_y = y[correct_answer]
        
        neg_y_hat = y_hat[wrong_answer]
        neg_y = y[wrong_answer]
        
        proxy_loss = loss_fn(sim_dist[correct_answer], pos_y)
        
        correct_sim_dist = sim_dist[correct_answer, pos_y]
        correct_ids = torch.argsort(correct_sim_dist)
        
        wrong_sim_dist = sim_dist[wrong_answer, neg_y]
        wrong_ids = torch.argsort(wrong_sim_dist)
        
        ##################################################
        w = torch.Tensor(np.array(list(range(1, len(correct_ids)+1))) / len(correct_ids)).cuda()
        pos_y_hat = pos_y_hat[correct_ids]
        pos_y = pos_y[correct_ids]
        
        pos_loss = 0
        for wi in range(len(w)):
            pos_loss += torch.nn.functional.cross_entropy(pos_y_hat[wi:wi+1, ], pos_y[wi:wi+1]) * w[wi:wi+1]
        
        w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
        neg_y_hat = neg_y_hat[wrong_ids]
        neg_y = neg_y[wrong_ids]
        
        neg_loss = 0
        for wi in range(len(w)):
            neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w[wi:wi+1]
        
            
        return (pos_loss + neg_loss) / 2. + proxy_loss
    
