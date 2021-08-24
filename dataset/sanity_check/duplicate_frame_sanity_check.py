import cv2
import os
import glob
import pickle
import numpy as np
from pandas import DataFrame as df

import shutil
import natsort
import matplotlib.pyplot as plt
from itertools import groupby


frame_path = '/OOB_RECOG/dataset/check_duplicate_frame/08092'
pickle_save_path = '/OOB_RECOG/0727/01_G_01_L_310_xx0_01_vsync1MpdecVideo_vsync1Cutting22.pkl'

def video_sanity_check(frame_path, pickle_save_path):
    frame_list = glob.glob(os.path.join(frame_path, '*'))
    frame_list = natsort.natsorted(frame_list)

    duplicatation_frame_sanity_list = []
    
    # check frame [0, 1] [1, 2] [2, 3] [3, 4]
    for fnum in range(0, len(frame_list)-2):

        frame1_idx = fnum
        frame2_idx = fnum+1

        output_1 = cv2.imread(frame_list[frame1_idx])
        output_2 = cv2.imread(frame_list[frame2_idx])

        mean = (output_2 - output_1).mean()
        diff = output_2 - output_1

        print(frame1_idx, mean)

        cv2.imwrite('diff-'+str(frame1_idx)+'.jpg', diff)

        if mean == 0.0:
            duplicatation_frame_sanity_list.append(1)
            print(f'\t ====> FRAME DUPLICATE : {frame1_idx}')
        else:
            duplicatation_frame_sanity_list.append(0)
    
    # list 저장 (pickle)
    with open(pickle_save_path, 'wb') as f:
        pickle.dump(duplicatation_frame_sanity_list, f)

    print('duplicatation_frame_sanity_list : ', duplicatation_frame_sanity_list)
    print('len(duplicatation_frame_sanity_list) : ', len(duplicatation_frame_sanity_list))

    # 저장 폴더 삭제
    shutil.rmtree(frame_path)

# ######## VISUALIZATION ########
def encode_list(s_list): # run-length encoding from list
	return_list = []
	temp_list = []
	for key, group in groupby(s_list):
		if key != 0:
			length = len(list(group)) * 100
		else:
			length = len(list(group))

		return_list.append([length, key])

	return return_list

# def encode_list(s_list): # run-length encoding from list
#     return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

def visual_duplicate_frame(pickle_save_path):
    with open(pickle_save_path, 'rb') as f:
        origin_data = pickle.load(f)

    print(origin_data)

    # predict_data = {'origin': origin_data}
    # predict_data = {'trans': trans_data}

    # yticks = ['origin', 'trans']
    
    predict_data = {}
    predict_data['duplicate'] = origin_data
    '''
	predict_data = {'origin': [1, 1, 1, 0, 0, 1, 1, 1],
	        'trans': [0, 0, 1, 1, 0, 0, 1, 1]
			}
	'''

    #### 1. matplotlib의 figure 및 axis 설정
    fig, ax = plt.subplots(1,1,figsize=(26,7)) # 1x1 figure matrix 생성, 가로(18인치)x세로(20인치) 크기지정
    height = 0.5 # bar chart thic
    colors = ['cadetblue', 'orange']
    label_names = ['duplicate X', 'duplicate O']

    encode_data = {}
    # for y_name in yticks:
    encode_data['duplicate'] = df(data=encode_list(predict_data['duplicate']), columns=['duplicate', 'value']) # [length, value]	

    # arrange data
    runlength_df = df(range(0,0)) # empty df
    # for y_name in yticks:
    runlength_df = runlength_df.append(encode_data['duplicate'])

    # Nan -> 0, convert to int
    runlength_df = runlength_df.fillna(0).astype(int)

    # split data, class // both should be same length
    runlength_class = runlength_df['value'] # class info
    runlength_model = runlength_df['duplicate'] # run length info of model prediction

    # data processing for barchart
    data = np.array(runlength_model.to_numpy()) # width
    data_cum = data.cumsum(axis=0) # for calc start index

    print(data)
    print(data_cum)

    #### 2. bar 그리기
    for i, frame_class in enumerate(runlength_class) :
        widths = data[i]
        starts= data_cum[i] - widths
        
        bar = ax.barh(range(1), widths, left=starts, height=height, color=colors[frame_class]) # don't input label

    #### 3. title 설정
    fig.suptitle('LAPA VIDEO TRANSCODING', fontsize=16)
    ax.set_title('FFMPEG | MPDECIMATE')

    #### 4. 범례 나타내기
    box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

    plt.savefig('/OOB_RECOG/0727/01_G_01_L_310_xx0_01_vsync1MpdecVideo_vsync1Cutting.png', format='png', dpi=500)


if __name__ == '__main__':
    video_sanity_check(frame_path=frame_path, pickle_save_path=pickle_save_path)
    visual_duplicate_frame(pickle_save_path=pickle_save_path)
