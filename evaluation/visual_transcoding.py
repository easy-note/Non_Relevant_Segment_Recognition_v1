import os

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df
from itertools import groupby

import pickle

def encode_list(s_list): # run-length encoding from list
	return_list = []
	temp_list = []
	for key, group in groupby(s_list):
		# print('key ', key)
		# print('group', group)
		if key != 0:
			length = len(list(group)) * 100
		else:
			length = len(list(group))

		return_list.append([length, key])

	return return_list

# def encode_list(s_list): # run-length encoding from list
#     return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

def main():
	#### 2. matplotlib의 figure 및 axis 설정
	fig, ax = plt.subplots(1,1,figsize=(26,7)) # 1x1 figure matrix 생성, 가로(18인치)x세로(20인치) 크기지정
	height = 0.5 # bar chart thic
	colors = ['cadetblue', 'orange', 'green']
	label_names = ['duplicate X']

	with open('list_2997.pkl', 'rb') as f:
		data = pickle.load(f)

	encode_data = {}
	encode_data['duplicated'] = df(data=encode_list(data), columns=['length', 'value']) # [length, value]	

	runlength_df = df(range(0,0)) # empty df
	runlength_df = runlength_df.append(encode_data['duplicated'])
	runlength_df = runlength_df.fillna(0).astype(int)


	# split data, class // both should be same length
	runlength_class = runlength_df['value'] # class info
	runlength_model = runlength_df['length'] # run length info of model prediction

	# data processing for barchart
	data = np.array(runlength_model.to_numpy()) # width
	data_cum = data.cumsum(axis=0) # for calc start index

	print(data)
	print(data_cum)

	#### 3. bar 그리기
	for i, frame_class in enumerate(runlength_class) :
		widths = data[i]
		starts= data_cum[i] - widths
		
		bar = ax.barh(range(1), widths, left=starts, height=height, color=colors[frame_class]) # don't input label
	
	#### 4. title 설정
	fig.suptitle('LAPA VIDEO TRANSCODING', fontsize=16)
	ax.set_title('fps: 29.97')
	
	#### 8. 범례 나타내기
	box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
	ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
	ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)
	
	plt.savefig('lapa2997-2.png', format='png', dpi=500)


if __name__=='__main__':
	main()

	