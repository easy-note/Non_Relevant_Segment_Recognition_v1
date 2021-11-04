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
        self.FN_CLASS, self.FP_CLASS = (2,3)
        self.visual_helper = VisualHelper()
        self.EXCEPTION_NUM = -100

        self.gt_list = gt_list
        self.patient_name = patient_name
        self.save_path = save_path # "~.png"

    def _get_plt(self, visual_type):
        assert visual_type in ['predict', 'hem'], 'NOT SUPPORT VISUAL MODE'

        if visual_type == 'predict':
            fig, ax = plt.subplots(3,1,figsize=(18,15)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

            plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0.35)

        elif visual_type == 'hem':
            fig, ax = plt.subplots(1,1,figsize=(26,20)) # 1x1 figure matrix 생성, 가로(18인치)x세로(20인치) 크기지정

        return fig, ax
        
    def set_gt_list(self, gt_list):
        self.gt_list = gt_list

    def set_patient_name(self, paitent_name):
        self.patient_name = patient_name

    # for text on bar
    def present_text(self, ax, bar, text, color='black'):
        for rect in bar:
            posx = rect.get_x()
            posy = rect.get_y() - rect.get_height()*0.1
            print(posx, posy)
            ax.text(posx, posy, text, color=color, rotation=0, ha='left', va='bottom')

    # for section metric
    def draw_plot(self, ax, title, x_value, y_value, y_min, y_max, color='blue'):
        ax.plot(x_value, y_value, marker='o', markersize=4, alpha=1.0, color=color)

        # set title
        ax.set_title(title)
        
        # set x ticks
        ax.set_xticks(x_value)
        xtick_labels = ['{}'.format(frame) if i_th % 2 == 0 else '\n{}'.format(frame) for i_th, frame in enumerate(x_value)]
        ax.set_xticklabels(xtick_labels) # xtick change
        ax.xaxis.set_tick_params(labelsize=6)
        ax.set_xlabel('Start Frame', fontsize=12)
        
        # set y ticks
        ax.set_ylim(y_min, y_max)

        # 보조선
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

    def visual_predict(self, predict_list, model_name, inference_interval, window_size = 100, section_num = 2):
        # TO-DO: if call gt_list from self.patient_name, better
        # TO-DO: visualization oob ratio, metrics on ax
        '''
        predict_list: [1,0,0...]
        model_name: str - 'mobilenet .. ' => title name
        inferene_interval: int
        inference_interval: number of elements in each section => title name
        section number: number of elements in each section => title name
        '''
        
        # calc section metric per model

        gt_list, predict_list = self.visual_helper.fit_to_min_length(self.gt_list, predict_list)
        metrics = self.visual_helper.calc_metrics_with_index(gt_list, predict_list)
        metrics_per_section = self.visual_helper.calc_section_metrics(gt_list, predict_list, window_size, section_num) # not yet used
        
        fig, ax = self._get_plt('predict')
        
        ### for plt variable, it should be pair synk
        label_names = ['RS', 'NRS', 'FN', 'FP']
        colors = ['cadetblue', 'orange', 'blue', 'red']
        height = 0.5 # bar chart thic

        ### prepare data ###
        '''
        visual_data = {'GT': [1, 1, 1, 0, 0, 1, 1, 1], # 0번 ~ 2번 frame , 5번 ~ 7번 frame
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

        visual_data = {
            'GT':gt_list,
            'PREDICT':predict_list,
        }

        # for visualize FP, FN // change class
        predict_df = pd.DataFrame(predict_list)
        predict_df.iloc[metrics['FN_idx'],] = self.FN_CLASS
        predict_df.iloc[metrics['FP_idx'],] = self.FP_CLASS
        visual_data['PREDICT'] = list(np.array(predict_df[0].tolist()))

        encode_data = {}
        for y_name in yticks : # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.visual_helper.encode_list(visual_data[y_name]), columns=[y_name, 'class']) # [length, value]

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
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.FN_CLASS], height=height, color=colors[self.FN_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.FP_CLASS], height=height, color=colors[self.FP_CLASS]) # dummy data
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

        # write text on PREDICT bar
        gt_oob_ratio = metrics['gt_OOB'] / (metrics['gt_IB'] + metrics['gt_OOB'])
        predict_oob_ratio = metrics['predict_OOB'] / (metrics['predict_IB'] + metrics['predict_OOB'])
        text_bar = ax[0].barh(1, 0, height=height) # dummy data        
        self.present_text(ax[0], text_bar, 'CR : {:.3f} | OR : {:.3f} | JACCARD: {:.3f} | OOB_RATIO(GT) : {:.2f} | OOB_RATIO(PD) : {:.2f}'.format(metrics['CR'], metrics['OR'], metrics['Jaccard'], gt_oob_ratio, predict_oob_ratio))

        ### write on figure 
        # set title
        title_name = 'Predict of {}'.format(self.patient_name)
        sub_title_name = 'model: {} | inferene interval: {} | windows size: {} | section num: {}'.format(model_name, inference_interval, window_size, section_num)
        fig.suptitle(title_name, fontsize=16)
        ax[0].set_title(sub_title_name)

        # set xticks pre section size
        ax[0].set_xticks(range(0, len(frame_label), window_size))
        
        xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]
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

        # 10. draw subplot ax (section metrics)
        section_start_idx = np.array(metrics_per_section['start_idx']) * inference_interval
        section_start_idx = section_start_idx.tolist()
        CR_value = metrics_per_section['section_CR']
        OR_value = metrics_per_section['section_OR']

        CR_value = [1.0 if val==self.EXCEPTION_NUM else val for val in CR_value] # -100 EXP 일 경우 1로 처리
        OR_value = [0.0 if val==self.EXCEPTION_NUM else val for val in OR_value] # -100 EXP 일 경우 0로 처리
        
        self.draw_plot(ax[1], 'CR of Predict', section_start_idx, CR_value, y_min=-1.0, y_max=1.0, color='blue')
        self.draw_plot(ax[2], 'OR of Predict', section_start_idx, OR_value, y_min=0.0, y_max=1.0, color='red')

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

        return decode_list

    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

    def idx_to_time(self, idx, fps) :
        time_s = idx // fps
        frame = int(idx % fps)

        converted_time = str(datetime.timedelta(seconds=time_s))
        converted_time = converted_time + ':' + str(frame)

        return converted_time

    def calc_section_metrics(self, gt_list, predict_list, window_size, section_num):
        metrics_per_section = {
            'start_idx':[],
            'end_idx':[],
            'section_CR':[],
            'section_OR':[],
        }

        data = {
            'GT': gt_list,
            'PREDICT': predict_list,
        }

        total_info_df = pd.DataFrame(data)
        total_len = len(total_info_df)
        slide_window_start_end_idx= [[start_idx * window_size, (start_idx + section_num) * window_size] for start_idx in range(int(total_len/window_size))] # overlapping section

        # calc metric per section
        for start, end in slide_window_start_end_idx : # slicing            
            section_df = total_info_df.iloc[start:end, ]

            section_gt_list, section_predict_list = section_df['GT'].tolist(), section_df['PREDICT'].tolist()
            section_metrics = self.calc_metrics_with_index(section_gt_list, section_predict_list)
            
            end = start + len(section_df) - 1 # truly end idx

            metrics_per_section['start_idx'].append(start)
            metrics_per_section['end_idx'].append(end)
            metrics_per_section['section_CR'].append(section_metrics['CR'])
            metrics_per_section['section_OR'].append(section_metrics['OR'])

        return metrics_per_section
        
    def calc_metrics_with_index(self, gt_list, predict_list):
        metric_helper = MetricHelper()
        metric_helper.write_preds(np.array(predict_list), np.array(gt_list))
        metrics = metric_helper.calc_metric()
        
        # with metric index
        gt_np = np.array(gt_list)
        predict_np = np.array(predict_list)

        metrics['TP_idx'] = np.where((predict_np == self.NRS_CLASS) & (gt_np == self.NRS_CLASS))[0].tolist()
        metrics['TN_idx'] = np.where((predict_np == self.RS_CLASS) & (gt_np == self.RS_CLASS))[0].tolist()
        metrics['FP_idx'] = np.where((predict_np == self.NRS_CLASS) & (gt_np == self.RS_CLASS))[0].tolist()
        metrics['FN_idx'] = np.where((predict_np == self.RS_CLASS) & (gt_np == self.NRS_CLASS))[0].tolist()

        return metrics