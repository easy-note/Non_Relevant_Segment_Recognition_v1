import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as df

from itertools import groupby

import argparse
import json


parser = argparse.ArgumentParser()

parser.add_argument('--title_name', type=str, default='R017', help='trained model_path')

parser.add_argument('--GT_path', type=str, default='./new_results-robot_oob_resnet50-1_3-last/R017/R017_ch1_video_01/Inference-R017_ch1_video_01.csv', help='trained model_path')

parser.add_argument('--model_name', type=str, nargs='+', default=["resnet50", "wide_resnet50_2"],
					choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d', 'mobilenet_v2', 'mobilenet_v3_small', 'squeezenet1_0'], help='trained backborn model')

parser.add_argument('--model_infernce_path', type=str, nargs='+',
					default=["./new_results-robot_oob_resnet50-1_3-last/R017/R017_ch1_video_01/Inference-R017_ch1_video_01.csv", "./new_results-robot_oob_wide_resnet50_2-1_3-last/R017/R017_ch1_video_01/Inference-R017_ch1_video_01.csv"], help='inference video')

parser.add_argument('--results_save_dir', type=str, default= './visual_results', help='inference results save path')

args, _ = parser.parse_known_args()

def encode_list(s_list): # run-length encoding from list
    return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]



def main():

	print(args)
	print(json.dumps(args.__dict__, indent=2))


	#### 1. bar plot으로 나타낼 데이터 입력
	IB, OOB = (0,1) # class index
	
	### for plt variable, it should be pair sink
	label_names = ['IB', 'OOB']
	colors = ['cadetblue', 'orange']
	height = 0.5 # bar chart thic

	## Data prepare
	GT_df = pd.read_csv(args.GT_path)
	frame_label = GT_df['frame'] # sync xticks label from GT

	yticks = ['GT'] # y축 names # 순서중요
	yticks += args.model_name

	predict_data = {'GT': GT_df['truth']}
	
	# pairwise read
	for y_name, inf_path in zip(args.model_name, args.model_infernce_path) :
		predict_data[y_name] = pd.read_csv(inf_path)['predict']

	print(predict_data)
	
	'''
	data = {'GT': [1, 1, 1, 0, 0, 1, 1, 1], # 0번 ~ 2번 frame , 5번 ~ 7번 frame
	        'Model A': [0, 0, 1, 1, 0, 0, 1, 1], # 2번 ~ 3번 frame
			'Model B': [1, 0, 1, 1, 1, 0, 1, 0] # 0번 ~ 0번 frame, 2번 ~ 4번 frame, 6번 ~ 6번 frame
			}
	'''

	## Data preprocessing

	# run-length encoding
	encode_data = {}
	for y_name in yticks :
		encode_data[y_name] = df(data=encode_list(predict_data[y_name]), columns=[y_name, 'class']) # [length, value]

	print(encode_data)
	
	# arrange data
	runlength_df = df(range(0,0)) # empty df
	for y_name in yticks :
		runlength_df = runlength_df.append(encode_data[y_name])

	# Nan -> 0, convert to int
	runlength_df = runlength_df.fillna(0).astype(int)
	print(runlength_df)	

	'''
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

	# split data, class // both should be same length
	runlength_class = runlength_df['class'] # class info
	runlength_model = runlength_df[yticks] # run length info of model prediction

	print(runlength_class)
	print(runlength_model)

	#### 2. matplotlib의 figure 및 axis 설정
	fig, ax = plt.subplots(1,1,figsize=(10,5)) # 1x1 figure matrix 생성, 가로(7인치)x세로(5인치) 크기지정
	
	##### initalize label for legned, this code should be write before writing barchart #####
	init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[IB], height=height, color=colors[IB]) # dummy data
	init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[OOB], height=height, color=colors[OOB]) # dummy data
	##### #### #### #### ##### #### #### #### 

	# data processing for barchart
	data = np.array(runlength_model.to_numpy()) # width
	data_cum = data.cumsum(axis=0) # for calc start index

	print(data)
	print(data_cum)

	#### 3. bar 그리기
	for i, frame_class in enumerate(runlength_class) :
		print(data[i,:])
		print(frame_class)

		widths = data[i,:]
		starts= data_cum[i,:] - widths
		
		bar = ax.barh(range(len(yticks)), widths, left=starts, height=height, color=colors[frame_class]) # don't input label
	
	#### 4. x축 세부설정
	ax.set_xticks(range(27444))
	ax.set_xticklabels(frame_label) # xtick change
	ax.xaxis.set_tick_params(labelsize=10)
	ax.set_xlabel('Frame', fontsize=14)
	
	#### 5. y축 세부설정
	ax.set_yticks(range(len(yticks)))
	ax.set_yticklabels(yticks, fontsize=10)	
	ax.set_ylabel('Model', fontsize=14)
	
	#### 6. 범례 나타내기
	box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
	ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
	ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)
	
	#### 7. 보조선(눈금선) 나타내기
	ax.set_axisbelow(True)
	ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
	
	#### 8. 그래프 저장하고 출력하기
	plt.savefig('./visual_results/ex_barhplot.png', format='png', dpi=300)
	plt.show()


if __name__=='__main__':
	main()



'''
N, K = 4, 3
data = np.random.rand(N, K)
tick_labels = ["a", "b", "c", "d"]
labels = ["x", "y", "z"]

normalized = data / data.sum(axis=1, keepdims=True)
cumulative = np.zeros(N)
tick = np.arange(N)


print('data\n',data)
print('cumulative\n', cumulative)
print('tick', tick)
print('normlaizaed\n', normalized)

for k in range(K):
    print('-----')
    print('k', k)
    color = plt.cm.viridis(float(k) / K, 1)
    print('color', color)
    plt.barh(tick, normalized[:, k], left=cumulative, color=color, label=labels[k])
    # plt.bar(tick, normalized[:, k], bottom=cumulative, color=color, label=labels[k])
    print('cum', cumulative)
    cumulative += normalized[:, k]
    print('cum_add', cumulative)

plt.xlim((0, 1))
# plt.ylim((0, 1))
plt.yticks(tick, tick_labels)
# plt.xticks(tick, tick_labels)
plt.legend()
# plt.show()
plt.savefig('./visual_results/temp_frame_visual.png', dpi=200)
'''

