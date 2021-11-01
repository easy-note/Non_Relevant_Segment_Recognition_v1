import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from itertools import groupby

from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)

class VisualTool:
    def __init__(self, gt_list, patient_name, save_path):
        self.RS_CLASS, self.NRS_CLASS = (0,1)
        self.visual_helper = VisualHelper()

        self.gt_list = gt_list
        self.patient_name = patient_name
        self.save_path = save_path # "~.png"

    def _get_plt(self, visual_type):
        assert visual_type in ['predict', 'hem'], 'NOT SUPPORT VISUAL MODE'

        if visual_type == 'predict':
            fig, ax = plt.subplots(3,1,figsize=(26,20)) # 1x1 figure matrix 생성, 가로(18인치)x세로(20인치) 크기지정
        elif visual_type == 'hem':
            fig, ax = plt.subplots(1,1,figsize=(26,20)) # 1x1 figure matrix 생성, 가로(18인치)x세로(20인치) 크기지정

        return fig, ax
        
    def set_gt_list(self, gt_list):
        self.gt_list = gt_list

    def set_patient_name(self, paitent_name):
        self.patient_name = patient_name

    def visual_predict(self, predict_list, model_name, inference_interval):
        # TO-DO: if call gt_list from self.patient_name, better
        # TO-DO: visualization oob ratio, metrics on ax

        gt_list, predict_list = self.visual_helper.fit_to_min_length(self.gt_list, predict_list)
        metrics = self.visual_helper.calc_metric(gt_list, predict_list)
        calc_index = self.visual_helper.get_calc_index(gt_list, predict_list)

        print('\nmetrics')
        print(metrics)

        print('\n\ncalc_index')
        print(calc_index)
        
        fig, ax = self._get_plt('predict')
        
        ### for plt variable, it should be pair synk
        label_names = ['RS', 'NRS']
        colors = ['cadetblue', 'orange']
        height = 0.5 # bar chart thic

        ### prepare data ###
        '''
        predict_data = {'GT': [1, 1, 1, 0, 0, 1, 1, 1], # 0번 ~ 2번 frame , 5번 ~ 7번 frame
	        'Model A': [0, 0, 1, 1, 0, 0, 1, 1], # 2번 ~ 3번 frame
			'Model B': [1, 0, 1, 1, 1, 0, 1, 0] # 0번 ~ 0번 frame, 2번 ~ 4번 frame, 6번 ~ 6번 frame
        }

	    runlength_df = df([[3,0,0,1],
				[2,0,0,0],
				[3,0,0,1],
				[0,2,0,0],
				[0,2,0,1],
				[0,2,0,0],
				[0,2,0,1],
				[0,0,1,1],
				[0,0,1,0],
				[0,0,3,1],
				[0,0,1,0],
				[0,0,1,1],
				[0,0,1,0]], columns= yticks + ['class'])
	    '''
        frame_label = list(range(0, len(gt_list) * inference_interval, inference_interval))
        time_label = [self.visual_helper.idx_to_time(idx, fps=30) for idx in frame_label]
        yticks = ['GT', 'PREDICT'] # y축 names, 순서중요

        predict_data = {
            'GT':gt_list,
            'PREDICT':predict_list,
        }
        encode_data = {}
        for y_name in yticks : # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.visual_helper.encode_list(predict_data[y_name]), columns=[y_name, 'class']) # [length, value]

        # arrange data
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        for y_name in yticks :
            runlength_df = runlength_df.append(encode_data[y_name])

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction

        ### draw ###
        ##### initalize label for legned, this code should be write before writing barchart #####
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.RS_CLASS], height=height, color=colors[self.RS_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NRS_CLASS], height=height, color=colors[self.NRS_CLASS]) # dummy data
	    ##### #### #### #### ##### #### #### #### 

        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        # draw bar
        for i, frame_class in enumerate(runlength_class) :
            # print(data[i,:])
            # print(frame_class)

            widths = data[i,:]
            starts= data_cum[i,:] - widths
            
            bar = ax[0].barh(range(len(yticks)), widths, left=starts, height=height, color=colors[frame_class]) # don't input label

        ### write figure 
        # set title
        title_name = 'Predict of {}'.format(self.patient_name)
        sub_title_name = 'model: {} | inferene interval: {}'.format(model_name, inference_interval)
        fig.suptitle(title_name, fontsize=16)
        ax[0].set_title(sub_title_name)

        # set xticks
        step_size = 30 # xtick step size : How many frame count in One section
        ax[0].set_xticks(range(0, len(frame_label), step_size)) # step_size
        
        xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::step_size], time_label[::step_size]))]
        ax[0].set_xticklabels(xtick_labels) # xtick change
        ax[0].xaxis.set_tick_params(labelsize=6)
        ax[0].set_xlabel('Frame / Time (h:m:s:fps)', fontsize=12)

        # set yticks
        ax[0].set_yticks(range(len(yticks)))
        ax[0].set_yticklabels(yticks, fontsize=10)	
        ax[0].set_ylabel('Model', fontsize=12)

        # 8. legend
        box = ax[0].get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax[0].legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        # 9. 보조선(눈금선) 나타내기
        ax[0].set_axisbelow(True)
        ax[0].xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

         ### file write
	    # fig.tight_layout() # subbplot 간격 줄이기
        plt.show()
        plt.savefig(self.save_path, format='png', dpi=500)
    
    def visual_hem(self, neg_hard_idx, pos_hard_idx, neg_vanila_idx, pos_vanila_idx):
        # TO-DO : visualization of sampleing (for hem example)

        # 1. parsing gt from self.patient name
        # 2. visualization of function parameter (hard example idx, vanila idx)
        # 3. save plt
        pass

class VisualHelper:
    def __init__(self):
        self.RS_CLASS, self.NRS_CLASS = (0,1)

    def fit_to_min_length(self, gt_list, predict_list):
        gt_len, predict_len = len(gt_list), len(predict_list)
        min_len = min(gt_len, predict_len)

        gt_list, predict_list = gt_list[:min_len], predict_list[:min_len]

        return gt_list, predict_list

    def decode_list(self, run_length): # run_length -> [0,1,1,1,1,0 ...]
        decode_list = []

        for length, group in run_length : 
            decode_list += [group] * length
        
        # print(decode_list)

        return decode_list

    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

    def idx_to_time(self, idx, fps) :
        time_s = idx // fps
        frame = int(idx % fps)

        converted_time = str(datetime.timedelta(seconds=time_s))
        converted_time = converted_time + ':' + str(frame)

        return converted_time


    def calc_metric(self, gt_list, predict_list):
        metric_helper = MetricHelper()
        metric_helper.write_preds(np.array(predict_list), np.array(gt_list))
        metrics = metric_helper.calc_metric()

        return metrics

    def get_calc_index(self, gt_list, predict_list):
        calc_index = {
            'TP':[],
            'TN':[],
            'FP':[],
            'FN':[],
        }

        gt_np = np.array(gt_list)
        predict_np = np.array(predict_list)

        calc_index['TP'] = np.where((predict_np == self.NRS_CLASS) | (gt_np == self.NRS_CLASS))[0].tolist()
        calc_index['TN'] = np.where((predict_np == self.RS_CLASS) | (gt_np == self.RS_CLASS))[0].tolist()
        calc_index['FP'] = np.where((predict_np == self.NRS_CLASS) | (gt_np == self.RS_CLASS))[0].tolist()
        calc_index['FN'] = np.where((predict_np == self.RS_CLASS) | (gt_np == self.NRS_CLASS))[0].tolist()

        return calc_index