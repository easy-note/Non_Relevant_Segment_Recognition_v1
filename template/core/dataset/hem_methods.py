import torch
import torch.nn as nn

import pandas as pd
from torch.utils.data import DataLoader


class HEMHelper():
    """
        Help computation ids for Hard Example Mining.
        
    """
    def __init__(self):
        super().__init__()
        
    def set_method(self, method):
        self.method = method

    def set_batch_size(self, bsz):
        self.bsz = bsz

    def set_n_batch(self, N):
        self.n_bs = N

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

        for i, data in enumerate(data_loader):
            img_path_list = list(data['img_path'])
            gt_list = list(data['y'].cpu().data.numpy())
            predict_list = list(data['y_hat'].cpu().data.numpy())
            logit_list = list(data['logit'].cpu().data.numpy())

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

        ### 1. 정답 맞춘 여부와 상관없이, diff 가 작은 경우
        df = df.sort_values(by=['Diff'], axis=0) # 올림차순 정렬

        hem_df_with_small_diff = df.iloc[:50,:]
        df = df.iloc[50:,:]

        ### 2. 정답을 틀렸고, diff 값 차이가 큰 경우
        correct_df = df[df['Consensus']==CORRECT]
        incorrect_df = df[df['Consensus']==INCORRECT]
        
        incorrect_df = incorrect_df.sort_values(by=['Diff'], axis=0, ascending=False) # 내림차순 정렬
        hem_df_with_big_diff = incorrect_df.iloc[:50, :]

        ### 3. hem_assets_df = hem_df_with_small_diff + hem_df_with_big_diff
        hem_assets_df = pd.concat([hem_df_with_small_diff, hem_df_with_big_diff])[['Img_path', 'GT']]
        hem_assets_df = hem_assets_df.sort_values(by='Img_path', axis=0, ignore_index=True)
        hem_assets_df.columns = ['img_path', 'class_idx']

        ### TODO 4. IB, OOB 비율 맞추기
        correct_df = correct_df.sort_values(by=['Img_path'], axis=0)
        incorrect_df = incorrect_df.iloc[50:, :].sort_values(by='Img_path', axis=0)

        df = pd.concat([correct_df, incorrect_df])
        df = df.sort_values(by='Img_path', axis=0, ignore_index=True)
        
        return hem_assets_df

    def hem_vi(self, model, dataset):
        pass

    def hem_batch_sampling(self, model, dataset):
        d_loader = DataLoader(dataset, 
                            batch_size=self.bsz, 
                            shuffle=True,
                            drop_last=True)
        n_pick = self.bsz // (self.n_bs * 2)

        y_hat = None
        y_true = None

        # model.eval()

        for _ in range(self.n_bs):
            x, y = next(iter(d_loader))
            x, y = x.cuda(), y.cuda()

            output = nn.functional.softmax(model(x), -1)
            pred_ids = torch.argmax(output, -1)
            pos_chk = pred_ids == y
            neg_chk = pred_ids != y

            # pos_output = x[pos_chk][:n_pick, ]
            # neg_output = x[neg_chk][:n_pick, ]

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
        
        # model.train()

        return y_hat, y_true
