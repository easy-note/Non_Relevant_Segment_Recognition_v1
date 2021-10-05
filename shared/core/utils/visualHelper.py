import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df
from itertools import groupby

import os
import imageio
import glob
import natsort

class visualHelper:
    def __init__(self, gt_list:list, mobile_predict_list:list, efficient_predict_list:list, save_dir:str, frame_path="dummy"):
        self.gt_list = gt_list
        self.mobile_predict_list = mobile_predict_list
        self.efficient_predict_list = efficient_predict_list
        self.save_dir= save_dir
        self.frame_path = frame_path

        # for save gif
        self.mobilenet_save_dir = os.path.join(self.save_dir, 'mobilenet')
        self.efficientnet_save_dir = os.path.join(self.save_dir, 'efficientnet')

    def convert_predict_list(self, gt_list, predict_list):
        FN, FP = (2, 3)

        gt_list = gt_list.copy() # list call by reference
        predict_list = predict_list.copy() # list call by reference

        for idx, (gt_label, predict_label) in enumerate(zip(gt_list, predict_list)):
            if gt_label != predict_label:
                if predict_label == 0: # FN
                    predict_list[idx] = FN
                elif predict_label == 1: # FP
                    predict_list[idx] = FP

        return predict_list


    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]


    def inference_visual(self):

        #### 1. bar plot으로 나타낼 데이터 입력
        IB, OOB = (0,1) # class index

        ### for plt variable, it should be pair sink
        label_names = ['TN (IB)', 'TP (OOB)', 'FP', 'FN']
        colors = ['cadetblue', 'orange', 'purple', 'red']
        height = 0.5

        '''
        predict_data = {'gt': [1, 1, 1, 0, 0, 1, 1, 1],
                'mobile_predict': [0, 0, 1, 1, 0, 0, 1, 1]
                'efficient_predict': [0, 1, 1, 1, 0, 0, 1, 1]
                }   

        '''

        predict_data = {}
        predict_data['gt'] = self.gt_list
        predict_data['mobile_predict'] = self.convert_predict_list(self.gt_list, self.mobile_predict_list)
        predict_data['efficient_predict'] = self.convert_predict_list(self.gt_list, self.efficient_predict_list)

        yticks = ['gt', 'mobile_predict', 'efficient_predict']

        encode_data = {}
        
        for y_name in yticks : # ['gt', 'mobile_predict', 'efficient_predict']
            # print(encode_list(predict_data[y_name]))
            encode_data[y_name] = df(data=self.encode_list(predict_data[y_name]), columns=[y_name, 'class']) # [length, value] , [[785, 0], [29, 1], [827, 0], [17, 1], [1718, 0], [18, 1], [601, 0], [18, 1], [568, 0]]
        
        print(encode_data['gt'])
        print(encode_data['mobile_predict'])
        print(encode_data['efficient_predict'])

        # arrange data
        runlength_df = df(range(0,0)) # empty df

        for y_name in yticks : # ['gt', 'mobile_predict', 'efficient_predict']
            runlength_df = runlength_df.append(encode_data[y_name]) # encode_data[y_name] -> df

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)
        print(runlength_df)	

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction

        '''
        runlength_class
                0     0
                1     1
                2     0
                3     1
                4     0
                5     1
                6     0
                7     1
                8     0
                0     0
                1     3
                2     1
                3     2
                4     0
                5     3
                6     0
                7     3
                8     1
                9     0
                10    3
                11    1
                12    2
                13    0
                14    3
                15    0
                16    3
                17    1
                18    0
                0     0
                1     3
                2     1
                3     2
                4     0
                5     3
                6     1
                7     0
                8     3
                9     1
                10    2
                11    0
                12    3
                13    0
                14    3
                15    1
                16    0

        runlength_model
                gt  mobile_predict  efficient_predict
            0    785               0                  0
            1     29               0                  0
            2    827               0                  0
            3     17               0                  0
            4   1718               0                  0
            5     18               0                  0
            6    601               0                  0
            7     18               0                  0
            8    568               0                  0
            0      0             784                  0
            1      0               1                  0
            2      0              28                  0
            3      0               1                  0
            4      0             112                  0
            5      0               1                  0
            6      0             713                  0
            7      0               1                  0
            8      0              17                  0
            9      0            1717                  0
            10     0               1                  0
            11     0              17                  0
            12     0               1                  0
            13     0               2                  0
            14     0              18                  0
            15     0             580                  0
            16     0               1                  0
            17     0              18                  0
            18     0             568                  0
            0      0               0                784
            1      0               0                  1
            2      0               0                 28
            3      0               0                  1
            4      0               0                826
            5      0               0                  1
            6      0               0                 17
            7      0               0               1717
            8      0               0                  1
            9      0               0                 17
            10     0               0                  1
            11     0               0                  2
            12     0               0                 18
            13     0               0                580
            14     0               0                  1
            15     0               0                 18
            16     0               0                568
        '''

        #### 2. matplotlib의 figure 및 axis 설정
        fig, ax = plt.subplots(1,1,figsize=(26,10)) # 1x1 figure matrix 생성, 가로(18인치)x세로(20인치) 크기지정
        ax.plot(label=label_names)

        plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0, 
                        hspace=0.35)

        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        print(data)
        print(data_cum)

        #### 3. bar 그리기
        for i, frame_class in enumerate(runlength_class) :
            print(data[i,:])
            print(frame_class)

            widths = data[i,:] # [785   0   0]
            starts= data_cum[i,:] - widths # [ 785    0    0] - [ 785    0    0]
            
            bar = ax.barh(range(len(yticks)), widths, left=starts, height=height, color=colors[frame_class]) # don't input label



        # #### 3. title 설정
        fig.suptitle('ViHUB-Pro (windows)', fontsize=20)
        # ax.set_title('FFMPEG | MPDECIMATE')

        #### 7. y축 세부설정
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=15)	
        ax.set_ylabel('Model', fontsize=12)

        # #### 4. 범례 나타내기
        box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        
        ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=2)

        leg = ax.get_legend()
        leg.legendHandles[0].set_color('cadetblue')
        leg.legendHandles[1].set_color('orange')
        leg.legendHandles[2].set_color('red')
        leg.legendHandles[3].set_color('purple')

        plt.savefig(os.path.join(self.save_dir, 'visual.png'), format='png', dpi=500)

    def generate_gif(self):
        # 전체 frame dir path
        total_frames = glob.glob(os.path.join(self.frame_path ,'*.jpg'))
        total_frames = natsort.natsorted(total_frames)

        # gif save dir
        os.makedirs(self.mobilenet_save_dir, exist_ok=True)
        os.makedirs(self.efficientnet_save_dir, exist_ok=True)

        ## mobile
        FP, FN, TP, TN = 0, 0, 0, 0
        FN_idx_list = []
        FP_idx_list = []
        for idx, (gt, predict) in enumerate(zip(self.gt_list, self.mobile_predict_list)):
            if gt == predict:
                pass

            elif gt != predict:
                if predict == 0: # FN
                    FN_idx_list.append(idx)

                elif predict == 1: # FP
                    FP_idx_list.append(idx)

                    
        print(FN_idx_list)
        print(FP_idx_list)

        FP_frame_list = []
        FN_frame_list = []
        for i, frame in enumerate(total_frames):
            if i in FP_idx_list:
                FP_frame_list.append(frame)

            if i in FN_idx_list:
                FN_frame_list.append(frame)

        FP_imgaes = []
        for image in FP_frame_list:
            FP_imgaes.append(imageio.imread(image))

        speed_sec = { 'duration' : 0.7}
        
        try :
            imageio.mimsave(os.path.join(self.mobilenet_save_dir, 'mobilenet_FP.gif'), FP_imgaes, **speed_sec)
        except :
            pass

        FN_imgaes = []
        for image in FN_frame_list:
            FN_imgaes.append(imageio.imread(image))

        speed_sec = { 'duration' : 0.7}
        
        try : # when non element in FN_images
            imageio.mimsave(os.path.join(self.mobilenet_save_dir, 'mobilenet_FN.gif'), FN_imgaes, **speed_sec)
        except :
            pass


        ## efficient
        FP, FN, TP, TN = 0, 0, 0, 0
        FN_idx_list = []
        FP_idx_list = []
        for idx, (gt, predict) in enumerate(zip(self.gt_list, self.efficient_predict_list)):
            if gt == predict:
                pass

            elif gt != predict:
                if predict == 0: # FN
                    FN_idx_list.append(idx)

                elif predict == 1: # FP
                    FP_idx_list.append(idx)

                    
        print(FN_idx_list)
        print(FP_idx_list)

        FP_frame_list = []
        FN_frame_list = []
        for i, frame in enumerate(total_frames):
            if i in FP_idx_list:
                FP_frame_list.append(frame)
            
            if i in FN_idx_list:
                FN_frame_list.append(frame)

        FP_imgaes = []
        for image in FP_frame_list:
            FP_imgaes.append(imageio.imread(image))

        speed_sec = { 'duration' : 0.7}
        try :
            imageio.mimsave(os.path.join(self.efficientnet_save_dir, 'efficientnet_FP.gif'), FP_imgaes, **speed_sec)
        except :
            pass


        FN_imgaes = []
        for image in FN_frame_list:
            FN_imgaes.append(imageio.imread(image))

        speed_sec = { 'duration' : 0.7}
        try :
            imageio.mimsave(os.path.join(self.efficientnet_save_dir, 'efficientnet_FN.gif'), FN_imgaes, **speed_sec)
        except :
            pass

