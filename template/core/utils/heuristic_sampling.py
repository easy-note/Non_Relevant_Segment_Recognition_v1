
from itertools import groupby, accumulate

import pandas as pd
import numpy as np


class HeuristicSampler():
    def __init__(self, assets_df):
        # img_path 기준 sort 된 assets_df
        self.assets_df = assets_df.reset_index(drop=True)

        self.class_idx_list = self.assets_df['class_idx'].tolist()
        self.class_idx_len = len(self.class_idx_list)

        print('\n\n[TOTAL GT] class_idx_len : {}\n\n'.format(self.class_idx_len))

        ##### 1. Calculate NRS (Non Related Surgery) start_idx, end_idx [[30, 38], [50, 100], [150, 157], ... ,[497679, 497828]] 
        nrs_start_end_idx_list = self.calc_nrs_idx()

        ##### 2. Select RS (Wise-Related Surgery) idx
        self.assets_df = self.extract_wise_rs_idx(nrs_start_end_idx_list)

        ##### 3. Set ratio
        self.final_assets = self.set_ratio()


        # ##### 4. Last processing
        # self.img_list = self.final_assets.img_path.tolist()
        # self.label_list = self.final_assets.class_idx.tolist()


    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

    def wise_rs_parity_check(self, wise_rs_list):
        return len(wise_rs_idx) == self.class_idx_len



    def calc_nrs_idx(self):
        ##### 1. Calculate NRS (Non Related Surgery) start_idx, end_idx [[30, 38], [50, 100], [150, 157], ... ,[497679, 497828]] 
        encode_data = self.encode_list(self.class_idx_list)
        encode_df = pd.DataFrame(data=encode_data, columns=['length', 'class']) # [length, value]
        
        # arrange data
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        runlength_df = runlength_df.append(encode_df)

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_gt = runlength_df['length'] # run length info of gt

        # data processing for barchart
        data = np.array(runlength_gt.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        runlength_df['accum'] = data_cum # accumulation of gt
        
        nrs_runlength_df = runlength_df[runlength_df['class'] == 1]

        nrs_runlength_df['start_idx'] = nrs_runlength_df['accum'] - nrs_runlength_df['length']
        nrs_runlength_df['end_idx'] = nrs_runlength_df['accum'] - 1

        start_idx_list = nrs_runlength_df['start_idx'].tolist()
        end_idx_list = nrs_runlength_df['end_idx'].tolist()

        nrs_start_end_idx_list = []
        for start_idx, end_idx in zip(start_idx_list, end_idx_list):
            nrs_start_end_idx_list.append([start_idx, end_idx])

        print('\n\n================== NRS_START_END_IDX_LIST (len:{}) ================== \n\n{}\n\n'.format(len(nrs_start_end_idx_list), nrs_start_end_idx_list))

        return nrs_start_end_idx_list
        
        
    def extract_wise_rs_idx(self, nrs_start_end_idx_list):
        ##### 2. Select RS (Wise-Related Surgery) idx
        wise_rs_idx = [False] * self.class_idx_len

        for nrs_idx in nrs_start_end_idx_list:
            nrs_start_idx = nrs_idx[0]
            nrs_end_idx = nrs_idx[1]

            start_end_gap = nrs_end_idx-nrs_start_idx

            if nrs_start_idx == 0: # nrs start idx == 0 이면, 그 이전의 프레임을 선택할 수 없음. 
                pass
            elif nrs_start_idx-start_end_gap < 0: # nrs start idx != 0 인데, gap 을 뺀 후가 0보다 작다면, 0 ~ nrs_start_idx select. 
                wise_rs_idx[0:nrs_start_idx] = [True]*start_end_gap
            else: 
                wise_rs_idx[nrs_start_idx-start_end_gap:nrs_start_idx] = [True]*start_end_gap
            
            if nrs_end_idx+1 == self.class_idx_len: # nrs end idx + 1 == self.class_idx_len  이면, self.class_idx_len 을 넘어선 프레임을 선택할 수 없음. 
                pass
            elif nrs_end_idx+start_end_gap+1 > self.class_idx_len: # nrs end idx + 1 != self.class_idx_len  인데, gap 을 추가한 후가 self.class_idx_len 보다 크다면, nrs_end_idx+1 ~ 끝까지 select.
                wise_rs_idx[nrs_end_idx+1:self.class_idx_len] = [True]*start_end_gap    
            else:
                wise_rs_idx[nrs_end_idx+1:nrs_end_idx+start_end_gap+1] = [True]*start_end_gap

        ## Parity check of wise_rs_idx, self.class_idx_len.
        if not self.wise_rs_parity_check:
            raise Exception("\nERROR: NOT MATCH BETWEEN len(wise_rs_idx), class_idx_len ====> len(wise_rs_idx): {}, class_idx_len: {}".format(len(wise_rs_idx, class_idx_len)))
        
        
        ##### 2-1. Consensus between wise_rs_idx(rs+nrs) and self.class_idx_list (rs)
        final_wise_rs_idx = []
        for wise_rs, gt in zip(wise_rs_idx, self.class_idx_list):
            if (wise_rs==True) and (gt==0):
                final_wise_rs_idx.append(True)
            else:
                final_wise_rs_idx.append(False)

        self.assets_df['wise_rs'] = final_wise_rs_idx

        print('==================== FINAL self.assets_df ====================')
        print(self.assets_df, '\n\n')

        # pd.set_option('display.max_row', None)
        print('==================== Wise RS ====================')
        print(self.assets_df[self.assets_df['wise_rs'] == True], '\n\n')

        return self.assets_df


    def set_ratio(self):
        assets_nrs_df = self.assets_df[self.assets_df['class_idx']==1]

        assets_wise_rs_df = self.assets_df[self.assets_df['wise_rs']==True]
        
        # self.assets_vanila_df = self.assets_df[(self.assets_df['wise_rs']==False) & (self.assets_df['class_idx']==0)].sample(n=len(self.assets_nrs_df)*3-len(self.assets_wise_rs_df), random_state=self.random_seed)
        assets_vanila_df = self.assets_df[(self.assets_df['wise_rs']==False) & (self.assets_df['class_idx']==0)].sample(n=len(assets_nrs_df)*3-len(assets_wise_rs_df))

        # self.assets_rs_df = pd.concat([self.assets_wise_rs_df, self.assets_vanila_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        assets_rs_df = pd.concat([assets_wise_rs_df, assets_vanila_df]).sample(frac=1).reset_index(drop=True)

        # self.final_assets = pd.concat([self.assets_nrs_df, self.assets_rs_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        final_assets = pd.concat([assets_nrs_df, assets_rs_df]).sample(frac=1).reset_index(drop=True)

        print('\nself.assets_nrs_df\n', assets_nrs_df)
        print('\nself.assets_rs_df\n', assets_rs_df)

        print('\nself.final_assets\n', final_assets)

        return final_assets
    

    # # return img, label
    # def __getitem__(self, index):
    #     img_path, label = self.img_list[index], self.label_list[index]

    #     img = Image.open(img_path)
    #     img = self.aug(img)

    #     return img_path, img, label


        



