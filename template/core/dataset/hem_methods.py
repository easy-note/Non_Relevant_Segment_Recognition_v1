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
        
    def set_method(self, method):
        self.method = method

    def set_batch_size(self, bsz):
        self.bsz = bsz

    def set_n_batch(self, N):
        self.n_bs = N

    def set_ratio(self, hem_ib_df, hem_oob_df, ib_df, oob_df):
        # df 정렬
        hem_ib_df = hem_ib_df.sort_values(by='Img_path')
        hem_oob_df = hem_oob_df.sort_values(by='Img_path')
        ib_df = ib_df.sort_values(by='Img_path')
        oob_df = oob_df.sort_values(by='Img_path')

        # HEM 표기
        hem_ib_df['HEM'] = [1]*len(hem_ib_df)
        hem_oob_df['HEM'] = [1]*len(hem_oob_df)
        ib_df['HEM'] = [0]*len(ib_df)
        oob_df['HEM'] = [0]*len(oob_df)

        try:
            oob_assets_df = oob_df.sample(n=len(hem_oob_df), replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, hem_oob 개
        except:
            oob_assets_df = oob_df.sample(frac=1, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, 전체 oob_df

        target_ib_assets_df_len = (len(hem_oob_df)+len(oob_assets_df))*self.args.IB_ratio - len(hem_ib_df)
        print('target_ib_assets_df_len ==> {}'.format(target_ib_assets_df_len))
        
        try:
            ib_assets_df = ib_df.sample(n=target_ib_assets_df_len, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, target_ib_assets_df_len 개
        except:
            if target_ib_assets_df_len <= 0: 
                ib_assets_df = ib_df.sample(frac=0, replace=False, random_state=self.args.random_seed)
                hem_ib_df = hem_ib_df.sample(n=(len(hem_oob_df)+len(oob_assets_df))*self.args.IB_ratio, replace=False, random_state=self.args.random_seed)
            else:
                ib_assets_df = ib_df.sample(frac=1, replace=False, random_state=10)

        print('========='* 10)
        print('\them_ib_df {}, hem_oob_df {}, ib_assets_df {}, oob_assets_df {}'.format(len(hem_ib_df), len(hem_oob_df), len(ib_assets_df), len(oob_assets_df)))
        print('========='* 10)

        final_oob_assets = pd.concat([hem_oob_df, oob_assets_df])[['Img_path', 'GT', 'HEM']]
        final_ib_assets = pd.concat([hem_ib_df, ib_assets_df])[['Img_path', 'GT', 'HEM']]

        # sort & shuffle
        final_assets = pd.concat([final_oob_assets, final_ib_assets]).sort_values(by='Img_path', axis=0, ignore_index=True)
        print('\t>>>> SORT final_assets HEAD\n', final_assets.head(20), '\n\n')

        final_assets = final_assets.sample(frac=1, random_state=self.args.random_seed).reset_index(drop=True)
        print('\t>>>> SHUFFLE final_assets HEAD\n', final_assets.head(20), '\n\n')
        
        final_assets.columns = ['img_path', 'class_idx', 'HEM']

        return final_assets


    def compute_hem(self, model, dataset):
        if self.method == 'hem-softmax':
            return self.hem_softmax_diff(model, dataset)
        elif self.method == 'hem-vi':
            return self.hem_vi(model, dataset)
        elif self.method == 'hem-bs':
            return self.hem_batch_sampling(model, dataset)
        else: # exception
            return None

    def hem_softmax_diff(self, model, dataset):
        # pd.options.display.max_columns = None
        pd.options.display.max_rows = None

        cols = ['Img_path', 'GT', 'Predict', 'Logit', 'Diff', 'Consensus']
        CORRECT, INCORRECT = (0,1)
        IB_CLASS, OOB_CLASS = (0,1)

        for i, data in enumerate(dataset):
            img_path_list = list(data['img_path'])
            gt_list = list(data['y'].cpu().data.numpy())
            predict_list = list(data['y_hat'].cpu().data.numpy())
            logit_list = nn.Softmax(dim=1)(data['logit']).cpu().data.numpy()

            diff_list = [abs(logit[0]-logit[1]) for logit in logit_list]
            consensus_list = [CORRECT if y==y_hat else INCORRECT for y, y_hat in zip(gt_list, predict_list)]

            '''
                df = {
                    img_path | gt | predict | logit | diff | consensus 
                }
            '''
            
            if i == 0: # init df
                df = pd.DataFrame([x for x in zip(img_path_list, gt_list, predict_list, logit_list, diff_list, consensus_list)],
                                    columns=cols)
            else:
                df = df.append([pd.Series(x, index=df.columns) for x in zip(img_path_list, gt_list, predict_list, logit_list, diff_list, consensus_list)], 
                                ignore_index=True)

 
        df = df.sort_values(by=['Diff'], axis=0) # 올림차순 정렬

        print('\t>>>> TOTAL df | len(df): {}\n{}\n\n'.format(len(df), df.head(20)))

        ### 1. 정답 맞춘 여부와 상관없이, diff 가 작은 경우 (threshold: 0.1)
        hem_df_with_small_diff = df[df['Diff'] < self.args.hem_softmax_min_threshold]
        df = df[df['Diff'] >= self.args.hem_softmax_min_threshold]

        ### 2. 정답을 틀렸고, diff 값 차이가 큰 경우
        hem_df_with_big_diff = df[(df['Consensus']==INCORRECT) & (df['Diff'] > self.args.hem_softmax_max_threshold)]
        df = df[(df['Consensus']==CORRECT) | (df['Diff'] <= self.args.hem_softmax_max_threshold)]

        print('\t>>>> hem_df_with_small_diff | len(hem_df_with_small_diff): {}\n{}\n\n'.format(len(hem_df_with_small_diff), hem_df_with_small_diff.head(20)))
        print('\t>>>> hem_df_with_big_diff | len(hem_df_with_big_diff): {}\n{}\n\n'.format(len(hem_df_with_big_diff), hem_df_with_big_diff.head(20)))
        print('\t>>>> LEFT df | len(df): {}\n{}\n\n'.format(len(df), df.head(20)))


        ### 3. hem_assets_df = hem_df_with_small_diff + hem_df_with_big_diff
        hem_df = pd.concat([hem_df_with_small_diff, hem_df_with_big_diff])

        ### 4. set ib, oob ratio
        '''
            final_oob_assets : hem_oob_assets_df * 2 (hem_oob_assets_df 전체 포함 + alpha)
            final_ib_assets : final_oob_assets * 3 (hem_ib_assets_df 포함 + alpha)
        '''
        hem_ib_assets_df = hem_df[hem_df['GT']==IB_CLASS]
        hem_oob_assets_df = hem_df[hem_df['GT']==OOB_CLASS]
        
        ib_df = df[df['GT']==IB_CLASS]
        oob_df = df[df['GT']==OOB_CLASS]
        
        final_assets = self.set_ratio(hem_ib_assets_df, hem_oob_assets_df, ib_df, oob_df)

        print('\n\n','++++++++++'*10, '\nFINAL HEM ASSETS HEAD\n')
        print(final_assets.head(50))
        print('++++++++++'*10)

        return final_assets

    def hem_vi(self, model, dataset): # MC dropout
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
        # return df of final Hard Example
        '''

        return hem_df

    def hem_batch_sampling(self, model, dataset):
        d_loader = DataLoader(dataset, 
                            batch_size=self.bsz, 
                            shuffle=True,
                            drop_last=True)
        n_pick = self.bsz // (self.n_bs * 2)
        
        y_hat = None
        y_true = None

        for _ in range(self.n_bs):
            _, x, y = next(iter(d_loader))
            x, y = x.cuda(), y.cuda()

            output = nn.functional.softmax(model(x), -1)
            pred_ids = torch.argmax(output, -1)
            pos_chk = pred_ids == y
            neg_chk = pred_ids != y

            p_out = output[pos_chk]
            n_out = output[neg_chk]

            if self.args.sampling_type == 1:
                p_diff = torch.argsort(torch.abs(p_out[:,0] - p_out[:,1]), -1, True)[:n_pick]
                n_diff = torch.argsort(torch.abs(n_out[:,0] - n_out[:,1]), -1, True)[:n_pick]

                pos_output = p_out[p_diff]
                neg_output = n_out[n_diff]
                pos_y = y[pos_chk][p_diff]
                neg_y = y[neg_chk][n_diff]
                
            elif self.args.sampling_type == 2:
                n_pick = self.bsz // (self.n_bs * 4)

                p_diff = torch.argsort(torch.abs(p_out[:,0] - p_out[:,1]), -1, True)[:n_pick]
                n_diff = torch.argsort(torch.abs(n_out[:,0] - n_out[:,1]), -1, True)[:n_pick]
    
                p_diff = torch.cat((p_diff, torch.argsort(torch.abs(p_out[:,0] - p_out[:,1]), -1, True)[-n_pick:]), -1)
                n_diff = torch.cat((n_diff, torch.argsort(torch.abs(n_out[:,0] - n_out[:,1]), -1, True)[-n_pick:]), -1)

                pos_output = p_out[p_diff]
                neg_output = n_out[n_diff]
                pos_y = y[pos_chk][p_diff]
                neg_y = y[neg_chk][n_diff]

            elif self.args.sampling_type == 3: 
                pos_output = output[pos_chk][:n_pick, ]
                neg_output = output[neg_chk][:n_pick, ]

                pos_y = y[pos_chk][:n_pick, ]
                neg_y = y[neg_chk][:n_pick, ]
            
            if y_hat is not None:
                y_hat = torch.cat((y_hat, pos_output), 0)
                y_hat = torch.cat((y_hat, neg_output), 0)
            else:
                y_hat = pos_output
                y_hat = torch.cat((y_hat, neg_output), 0)

            if y_true is not None:
                y_true = torch.cat((y_true, pos_y), 0)
                y_true = torch.cat((y_true, neg_y), 0)
            else:
                y_true = pos_y
                y_true = torch.cat((y_true, neg_y), 0)

        return y_hat, y_true
