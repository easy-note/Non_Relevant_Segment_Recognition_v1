import os
import pandas as pd
import numpy as np

import csv
from collections import defaultdict
from matplotlib import pyplot as plt


def over_estimation_ratio(total_fold):

    for fold in total_fold: #fold1, fold2, fold3

        ##########################################
        ########## 1. root_file_path 정의 #########
        ##########################################
        root_file_path = '/Users/jihyunlee/Desktop/robot_0423_Consensus3/'

        file_path = root_file_path + fold + '/'

        pd.set_option('display.float_format', '{:.1f}'.format)
       
        for root, dirs, files in os.walk(file_path):
           
            dirs.sort(key=str.lower)
            files.sort(key=str.lower)

            for fname in files:
                if 'Patient_Total_metric-results' in fname: # Patient_Total_metric-temp2_results-robot_oob-wide_resnet50_2-fold_2-last.csv * 4 개 (각 모델 별)
                    data = pd.read_csv(root + '/' + fname)

                    over_ratio_list = []

                    for i in range(len(data)):
                        over_ratio = int(data.iloc[i, 2]) / int((data.iloc[i, 2]) + int(data.iloc[i, 3]) + int(data.iloc[i, 4]))
                        over_ratio_list.append(over_ratio)
                    
                    data['over_ratio'] = over_ratio_list

                    output_path = file_path
                    # if not os.path.exists(output_path):
                    #     os.makedirs(output_path)

                    data.to_csv(root + '/' + fname, float_format = '%.10f', index=False)


def fold_inference(total_fold): #['fold1', 'fold2', 'fold3']
    """ 
    fold 별 inference 결과 정리
    """

    for fold in total_fold: #fold1, fold2, fold3

        ##########################################
        ########## 1. root_file_path 정의 #########
        ##########################################
        root_file_path = '/Users/jihyunlee/Desktop/robot_0423_Consensus3/'

        file_path = root_file_path + fold + '/'

        df = pd.DataFrame({'Model (Unit: %)':['mobilenet_v3_small', 'resnext50_32x4d','squeezenet1_0', 'wide_resnet50_2']})
        pd.set_option('display.float_format', '{:.1f}'.format)
       
        data_total = []
        total_list = []
        
        for root, dirs, files in os.walk(file_path):
           
            dirs.sort(key=str.lower)
            files.sort(key=str.lower)

            for fname in files:
                if 'Patient_Total_metric-results' in fname: # Patient_Total_metric-temp2_results-robot_oob-wide_resnet50_2-fold_2-last.csv * 4 개 (각 모델 별)
                    data = pd.read_csv(root + '/' + fname)

                    ##############################################
                    ############## 2. 소수점인지 확인 ################
                    ##############################################
                    data = (data.iloc[0:, 14]) * 100 
                    
                    data_list = data.tolist() #[17번, 18번, 19번, 20번, 21번]
                    # print(data_list)

                    data_total.append(data_list)


        for i in range(len(data_total[0])):
            tmp_list = []
            for i_data in range(len(data_total)):
                tmp_list.append(data_total[i_data][i])

            total_list.append(tmp_list) #[[17, 17, 17, 17], [18, 18, 18, 18], ... , [21, 21, 21, 21]]

        fold1 = ['R_017', 'R_022', 'R_116', 'R_208', 'R_303']
        fold2 = ['R_003', 'R_004', 'R_006', 'R_013', 'R_018']
        fold3 = ['R_007', 'R_010', 'R_019', 'R_056', 'R_074']

        if fold == 'fold1':
            fold_name = fold1
        elif fold == 'fold2':
            fold_name = fold2
        else:
            fold_name = fold3

        for i in range(len(fold_name)):
            df[fold_name[i]] = total_list[i]

        # mAP 계산
        mAP_total = []
        for i in range(len(df)):
            mAP_df = df.iloc[i, 1:]
            mAP_list = np.array(mAP_df.tolist())
            mAP_list = list(map(float, mAP_list))
            mAP = sum(mAP_list) / len(mAP_list)
            mAP_total.append(mAP)

        df['mAP'] = mAP_total

        # csv 파일 생성
        output_path = file_path
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        df.to_csv(output_path + fold + '_total_inference.csv', float_format = '%.1f', index=False)

    # 전체 fold 합치기
    inference(root_file_path)


def inference(file_path):
    file_path = file_path
    dataframe_list = []

    for root, dirs, files in os.walk(file_path):
           
        dirs.sort(key=str.lower)
        files.sort(key=str.lower)

        for fname in files:
            if ('total_inference' in fname) and ('fold' in fname): # Patient_Total_metric-temp2_results-robot_oob-wide_resnet50_2-fold_2-last.csv * 4 개 (각 모델 별)
                data = pd.read_csv(root + '/' + fname)
                data = data.iloc[:, 1:6]
                dataframe_list.append(data)
            
    df = pd.concat(dataframe_list, axis=1)
    df = df.sort_index(axis=1)

    df.insert(0, "Model (Unit: %)",['mobilenet_v3_small', 'resnext50_32x4d','squeezenet1_0', 'wide_resnet50_2'], True)

    # mAP 계산
    mAP_total = []
    for i in range(len(df)):
        mAP_df = df.iloc[i, 1:]
        mAP_list = np.array(mAP_df.tolist())
        mAP_list = list(map(float, mAP_list))
        mAP = sum(mAP_list) / len(mAP_list)
        mAP_total.append(mAP)

    df['mAP'] = mAP_total

    AP_tmp = AP_cal(file_path)
    AP_list = []
    
    for i in AP_tmp:
        AP_list.append(i * 100)

    #AP 계산
    df['AP'] = AP_list

    df = df.reindex(index=[3, 1, 0, 2])
    output_path = file_path
    df.to_csv(output_path + 'total_inference.csv', float_format = '%.1f', index=False)

    total_inference_visualization(output_path + 'total_inference.csv', output_path)


def AP_cal(file_path):
    file_path = file_path
    model = ['mobilenet_v3_small', 'resnext50_32x4d','squeezenet1_0', 'wide_resnet50_2']
    
    model_dict = defaultdict(list)
    AP_list = []

    for root, dirs, files in os.walk(file_path):
        dirs.sort(key=str.lower)
        files.sort(key=str.lower)

        for file in files:
            for m in model:
                FP_cal = []
                if (m in file) and ('Patient_Total_metric-' in file):
                    # print(file)
                    data = pd.read_csv(root + '/' + file)

                    FP = sum(data['FP'])
                    TP = sum(data['TP'])
                    FN = sum(data['FN'])
                    TN = sum(data['TN'])

                    FP_cal.append(FP)
                    FP_cal.append(TP)
                    FP_cal.append(FN)
                    FP_cal.append(TN)

                    model_dict[m].append(FP_cal)
    # print(model_dict)

    for key, value in model_dict.items():
        FP, TP, FN, TN = 0, 0, 0, 0
 
        for i in range(len(value)):
            FP += value[i][0]
            TP += value[i][1]
            FN += value[i][2]
            TN += value[i][3]

            AP = (FP) / (FP + TP + FN)
        AP_list.append(AP)
    
    # print(AP_list)

    return AP_list


def total_inference_visualization(file_path, output_path):
    file_path = file_path
    data = pd.read_csv(file_path)

    fold1 = ['R017', 'R022', 'R116', 'R208', 'R303']
    fold2 = ['R003', 'R004', 'R006', 'R013', 'R018']
    fold3 = ['R007', 'R010', 'R019', 'R056', 'R074']
    total_fold = fold1 + fold2 + fold3 
    total_fold.sort()

    x_values = total_fold
    markers = ['o', 'x', '^', '*']

    for m in range(len(data)):
        y_values = list(data.loc[m][1:16])
        plt.plot(x_values, y_values, marker = markers[m])

    plt.legend(['wide_resnet50_2', 'resnext50_21x4d', 'mobilenet_v3_small', 'squeezenet1_0'])

    plt.xlabel('Patient')
    plt.ylabel('New OOB Metric (Unit: %)')
    plt.title('TOTAL_New OOB Metric')

    plt.xticks(rotation=45)

    output_path = output_path
    plt.savefig(output_path + 'total_visualization.png')

    plt.clf()

    data_compare(output_path + 'total_inference.csv', output_path)


def data_compare(new_data_path, output_path):
    old_data_path = '/Users/jihyunlee/Desktop/robot_0423_Consensus/Total_inference/mAP/FOLD_total_inference.csv'
    new_data_path = new_data_path

    x_values = ['R017', 'R022', 'R116', 'R208', 'R303', 'R003', 'R004', 'R006', 'R013', 'R018', 'R007', 'R010', 'R019', 'R056', 'R074']
    x_values.sort()

    model = ['wide_resnet50_2', 'resnext50_32x4d', 'mobilenet_v3_small', 'squeezenet1_0']

    for i, m in enumerate(model):

        old_data = pd.read_csv(old_data_path)
        old_data = list(old_data.iloc[i, 1:16])

        new_data = pd.read_csv(new_data_path)
        new_data = list(new_data.iloc[i, 1:16])

        plt.plot(x_values, old_data, marker = 'o')
        plt.plot(x_values, new_data, marker = '*') 

        plt.legend(['old_data', 'new_data'])
        plt.xlabel('Patient')
        plt.ylabel('New OOB Metric (Unit: %)')
        plt.title(m)
        plt.xticks(rotation=45)

        output_path = output_path
        plt.savefig(output_path + m + '_compare_visualization.png')

        plt.clf()

    param_visu(new_data_path, output_path)


def param_visu(data_path, output_path):

    file_path = data_path
    data = pd.read_csv(file_path)

    param = [66838338, 22984002, 928162, 736450]
    mAP = data.iloc[:, 16].tolist() #### 열 바꾸기

    colors = ['salmon', 'orange', 'steelblue', 'pink']
    markers = ['o', 'x', '^', '*']
    labels = ['wide_resnet50_2', 'resnext50_21x4d', 'mobilenet_v3_small', 'squeezenet1_0']

    x_values = param
    y_mAP_values = mAP

    for i, label in enumerate(labels):
        plt.scatter(x_values[i], y_mAP_values[i], marker = markers[i], label = label, s=100)

    output_path = output_path

    
    plt.ylabel('mAP (Unit: %)')
    plt.xlabel('# of param')

    plt.title('Param visualization')
    plt.legend(loc = 'best')
    plt.savefig(output_path + 'param_individual.png')




if __name__ == '__main__':
    fold_inference(['fold1', 'fold2', 'fold3'])