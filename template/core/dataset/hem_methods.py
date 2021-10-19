import torch
import torch.nn as nn

import pandas as pd
from torch.utils.data import DataLoader


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

    def hem_vi(self, model, dataset):
        pass

    def hem_batch_sampling(self, model, dataset):
        d_loader = DataLoader(dataset, 
                            batch_size=self.bsz, 
                            shuffle=True,
                            drop_last=True)
        n_pick = self.bsz // (self.n_bs * 2)
        # n_pick = self.bsz // (self.n_bs * 4)

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
