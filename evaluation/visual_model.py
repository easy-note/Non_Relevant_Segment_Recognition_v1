import os
import pandas as pd
import numpy as np

import csv
from collections import defaultdict
from matplotlib import pyplot as plt

from natsort import natsorted, index_natsorted, order_by_index

import seaborn as sns

EXCEPTION_NUM = -100

def calc_OOB_Evaluation_metric(FN_cnt, FP_cnt, TN_cnt, TP_cnt) :
	base_denominator = FP_cnt + TP_cnt + FN_cnt	
	# init
	EVAL_metric = {
		'CONFIDENCE_metric' : EXCEPTION_NUM,
		'correspondence' : EXCEPTION_NUM,
		'UN_correspondence' : EXCEPTION_NUM,
		'OVER_estimation' : EXCEPTION_NUM,
		'UNDER_estimtation' : EXCEPTION_NUM,
		'FN' : FN_cnt,
		'FP' : FP_cnt,
		'TN' : TN_cnt,
		'TP' : TP_cnt,
		'TOTAL' : FN_cnt + FP_cnt + TN_cnt + TP_cnt
	}


	if base_denominator > 0 : # zero devision except check, FN == full
		EVAL_metric['CONFIDENCE_metric'] = (TP_cnt - FP_cnt) / base_denominator
		EVAL_metric['correspondence'] = TP_cnt /  base_denominator
		EVAL_metric['UN_correspondence'] = (FP_cnt + FN_cnt) /  base_denominator
		EVAL_metric['OVER_estimation'] = FP_cnt / base_denominator
		EVAL_metric['UNDER_estimtation'] = FN_cnt / base_denominator
	
	return EVAL_metric


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
    old_data_path = './results_v1_robot_oob-mobilenet_v3_large-fold_1-last/Patient_Total_metric-ROBOT-results_v1_robot_oob-mobilenet_v3_large-fold_1-last.csv'
    new_data_path = new_data_path

    x_values = ['R017', 'R022', 'R116', 'R208', 'R303', 'R003', 'R004', 'R006', 'R013', 'R018', 'R007', 'R010', 'R019', 'R056', 'R074']
    x_values = ['R_2' 'R_6' 'R_13' 'R_74' 'R_100' 'R_202' 'R_301']
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

    # param_visu(new_data_path, output_path)

def visual_metric_per_patients_ver1(patient_total_metric_csv_path, model_name, title, save_path):
    
    x_ticks = 'Patient' # per patinets => it will be sorting by this column ['Patient', 'Video_name']
    # set value column | target_col = ['Confidence_Ratio', 'Over_Ratio', 'Under_Ratio', 'Correspondence', 'Un_Correspondence']
    
    patient_total_metric_df = pd.read_csv(patient_total_metric_csv_path)
    # 0. set plt
    fig, ax1 = plt.subplots(figsize=(12,8), tight_layout=True)
    ax2 = ax1.twinx()
    
    #### Confidence
    # 1. sorting by x_ticks
    patient_total_metric_df = patient_total_metric_df.reindex(index=order_by_index(patient_total_metric_df.index, index_natsorted(patient_total_metric_df[x_ticks])))

    print(patient_total_metric_df)

    # 2. plot
    target_col = 'Confidence_Ratio'
    y_value = patient_total_metric_df[target_col] * 100
    line1 = ax1.plot(patient_total_metric_df[x_ticks], y_value, marker='o', color='green', label=target_col)

    # 3. change xticks
    #ax1.set_xticklabels(rotation=45)
    ax1.tick_params(axis='x', labelrotation=45)

    # 4. set label
    ax1.set_xlabel(x_ticks)
    ax1.set_ylabel(target_col + ' (Unit: %)')

    # 5. set title
    ax1.set_title(model_name, size='12')
    plt.suptitle(title, size='15')

    #### Over Estimation
    target_col = 'Over_Ratio'
    y_value = patient_total_metric_df[target_col] * 100
    # ax2.plot(patient_total_metric_df[x_ticks], y_value, marker='x', color='deeppink')
    line2 = ax2.bar(patient_total_metric_df[x_ticks], y_value, color='deeppink', alpha=0.5, label=target_col)

    target_col = 'Under_Ratio'
    y_value = patient_total_metric_df[target_col] * 100
    # ax2.plot(patient_total_metric_df[x_ticks], y_value, marker='x', color='deeppink')
    line3 = ax2.bar(patient_total_metric_df[x_ticks], y_value, color='orange', alpha=0.5, label=target_col)

    ax2.set_ylabel('{} | {} (Unit: %)'.format('Over_Ratio', 'Under_Ratio'))

    # 6. set legend 
    ax1.grid(True)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # 6. save
    plt.savefig(save_path)

    # 7. clear plt
    plt.clf()

def visual_metric_per_patients_ver2(patient_total_metric_csv_path, title, save_path):

    
    x_ticks = 'Patient' # per patinets => it will be sorting by this column ['Patient', 'Video_name']
    # set value column | target_col = ['Confidence_Ratio', 'Over_Ratio', 'Under_Ratio', 'Correspondence', 'Un_Correspondence']
    
    patient_total_metric_df = pd.read_csv(patient_total_metric_csv_path)
    # 0. set plt
    fig, ax1 = plt.subplots(figsize=(14,10), tight_layout=True)
    ax2 = ax1.twinx()
    

    
    #### Confidence
    # 1. sorting by x_ticks
    patient_total_metric_df = patient_total_metric_df.reindex(index=order_by_index(patient_total_metric_df.index, index_natsorted(patient_total_metric_df[x_ticks])))

    print(patient_total_metric_df)

    # 2. plot
    target_col = 'Confidence_Ratio'
    y_value = patient_total_metric_df[target_col] * 100
    line1 = ax1.plot(patient_total_metric_df[x_ticks], y_value, marker='o', color='green', label=target_col)

    # 3. change xticks
    #ax1.set_xticklabels(rotation=45)
    ax1.tick_params(axis='x', labelrotation=45)

    # 4. set label
    ax1.set_xlabel(x_ticks)
    ax1.set_ylabel(target_col + ' (Unit: %)')

    # 5. set title
    plt.suptitle(title, size='15')

    #### Over Estimation
    # sum of FP, FN
    FP_value = patient_total_metric_df['FP']
    FN_value = patient_total_metric_df['FN']
    TP_value = patient_total_metric_df['TP']
    TN_value = patient_total_metric_df['TN']

    False_value = FP_value + FN_value

    # ax2.plot(patient_total_metric_df[x_ticks], y_value, marker='x', color='deeppink')
    line2 = ax2.bar(patient_total_metric_df[x_ticks], FP_value / False_value, color='deeppink', alpha=0.5, label='FP')

    # ax2.plot(patient_total_metric_df[x_ticks], y_value, marker='x', color='deeppink')
    line3 = ax2.bar(patient_total_metric_df[x_ticks], FN_value / False_value, bottom=FP_value / False_value, color='orange', alpha=0.5, label='FN')

    for idx, (rect1, rect2) in enumerate(zip(line2, line3)):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        ax2.text(rect1.get_x() + rect1.get_width()/2., height1/2., FP_value[idx], ha='center', va='bottom', rotation=0) # FP
        ax2.text(rect2.get_x() + rect2.get_width()/2., height1 + height2/2., FN_value[idx], ha='center', va='bottom', rotation=0) # FN

    line4 = ax2.plot(patient_total_metric_df[x_ticks], patient_total_metric_df['Over_Ratio'], marker='o', color='blue', label='Over_Ratio')

    
    ax2.set_ylim(0, 1.2) # set ylim
    ax2.set_ylabel('{} | {} Normalized (Unit: %)'.format('FP', 'FN'))

    # Calc CR, OR
    EVAL_metric = calc_OOB_Evaluation_metric(FN_value.sum(), FP_value.sum(), TN_value.sum(), TP_value.sum())

    ##### table ######
    rows = ['TP', 'TN', 'Confidence_Ratio', 'Over_Ratio']

    cell_text = []
    for row in rows :
        
        data = list(patient_total_metric_df[row])
        data_sum = sum(data)
        data_mean = data_sum / len(data)

        if row in ('FP', 'FN', 'TP', 'TN'):
            cell_text.append([x for x in data + ['-', '-']])

        elif row == 'Confidence_Ratio':
            cell_text.append(['%0.3f' % (x) for x in data + [data_mean, EVAL_metric['CONFIDENCE_metric']]])
        elif row == 'Over_Ratio':
            cell_text.append(['%0.3f' % (x) for x in data + [data_mean, EVAL_metric['OVER_estimation']]])

        else :
            cell_text.append([x for x in data + [data_mean, data_sum]])
        
    print(cell_text)

    # visual of table
    the_table = ax1.table(cellText = cell_text,
                        rowLabels=['$\\bf%s$' % (x) for x in rows],
                        colLabels= list(patient_total_metric_df['Patient']) + ['mean Avg', 'Avg'],
                        loc='top')

    # 7. set legend 
    ax1.grid(True)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # 8. save
    plt.savefig(save_path, dpi=300)

    # 9. clear plt
    plt.clf()
    


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


###### PATINET CASE ########

LAPA_CASE = ['L_301', 'L_303', 'L_305', 'L_309', 'L_317', 'L_325', 'L_326', 'L_340', 'L_346', 'L_349', 'L_412', 'L_421', 'L_423', 'L_442', 'L_443',
                'L_450', 'L_458', 'L_465', 'L_491', 'L_493', 'L_496', 'L_507', 'L_522', 'L_534', 'L_535', 'L_550', 'L_553', 'L_586', 'L_595', 'L_605', 'L_607', 'L_625',
                'L_631', 'L_647', 'L_654', 'L_659', 'L_660', 'L_661', 'L_669', 'L_676', 'L_310', 'L_311', 'L_330', 'L_333', 'L_367', 'L_370', 'L_377', 'L_379', 'L_385',
                'L_387', 'L_389', 'L_391', 'L_393', 'L_400', 'L_402', 'L_406', 'L_408', 'L_413', 'L_414', 'L_415', 'L_418', 'L_419', 'L_427', 'L_428', 'L_430', 'L_433',
                'L_434', 'L_436', 'L_439', 'L_471', 'L_473', 'L_475', 'L_477', 'L_478', 'L_479', 'L_481', 'L_482', 'L_484', 'L_513', 'L_514', 'L_515', 'L_517', 'L_537',
                'L_539', 'L_542', 'L_543', 'L_545', 'L_546', 'L_556', 'L_558', 'L_560', 'L_563', 'L_565', 'L_568', 'L_569', 'L_572', 'L_574', 'L_575', 'L_577', 'L_580']

ROBOT_CASE = ['R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_10', 'R_13', 'R_14', 'R_15', 'R_17', 'R_18', 'R_19', 'R_22', 'R_48', 'R_56', 'R_74',
                'R_76', 'R_84', 'R_94', 'R_100', 'R_116', 'R_117', 'R_201', 'R_202', 'R_203', 'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_301',
                'R_302', 'R_303', 'R_304', 'R_305', 'R_310', 'R_311', 'R_312', 'R_313', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_336', 'R_338', 'R_339', 'R_340',
                'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357', 'R_358', 'R_362', 'R_363', 'R_369', 'R_372', 'R_376', 'R_378', 'R_379', 'R_386',
                'R_391', 'R_393', 'R_399', 'R_400', 'R_402', 'R_403', 'R_405', 'R_406', 'R_409', 'R_412', 'R_413', 'R_415', 'R_418', 'R_419', 'R_420', 'R_423', 'R_424',
                'R_427', 'R_436', 'R_445', 'R_449', 'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_526', 'R_532', 'R_533']


###### OOB VIDEO LIST ########
# ANNOTATION PATH - /NAS/DATA2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1 # /data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1
# ANNOTATION PATH - /NAS/DATA2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2 # /data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2

# 91ea, 40case # /NAS/DATA/HuToM/Video_Robot_cordname # /data1/HuToM/Video_Robot_cordname
OOB_robot_40 = [
    'R_1_ch1_01', 'R_1_ch1_03', 'R_1_ch1_06', 'R_2_ch1_01', 'R_2_ch1_03', 'R_3_ch1_01', 'R_3_ch1_03', 'R_3_ch1_05', 'R_4_ch1_01', 'R_4_ch1_04', 
    'R_5_ch1_01', 'R_5_ch1_03', 'R_6_ch1_01', 'R_6_ch1_03', 'R_6_ch1_05', 'R_7_ch1_01', 'R_7_ch1_04', 'R_10_ch1_01', 'R_10_ch1_03', 'R_13_ch1_01', 
    'R_13_ch1_03', 'R_14_ch1_01', 'R_14_ch1_03', 'R_14_ch1_05', 'R_15_ch1_01', 'R_15_ch1_03', 'R_17_ch1_01', 'R_17_ch1_04', 'R_17_ch1_06', 'R_18_ch1_01', 
    'R_18_ch1_04', 'R_19_ch1_01', 'R_19_ch1_03', 'R_19_ch1_05', 'R_22_ch1_01', 'R_22_ch1_03', 'R_22_ch1_05', 'R_48_ch1_01', 'R_48_ch1_02', 'R_56_ch1_01', 
    'R_56_ch1_03', 'R_74_ch1_01', 'R_74_ch1_03', 'R_76_ch1_01', 'R_76_ch1_03', 'R_84_ch1_01', 'R_84_ch1_03', 'R_94_ch1_01', 'R_94_ch1_03', 'R_100_ch1_01', 
    'R_100_ch1_03', 'R_100_ch1_05', 'R_116_ch1_01', 'R_116_ch1_03', 'R_116_ch1_06', 'R_117_ch1_01', 'R_117_ch1_03', 'R_201_ch1_01', 'R_201_ch1_03', 'R_202_ch1_01', 
    'R_202_ch1_03', 'R_202_ch1_05', 'R_203_ch1_01', 'R_203_ch1_03', 'R_204_ch1_01', 'R_204_ch1_02', 'R_205_ch1_01', 'R_205_ch1_03', 'R_205_ch1_05', 'R_206_ch1_01', 
    'R_206_ch1_03', 'R_207_ch1_01', 'R_207_ch1_03', 'R_208_ch1_01', 'R_208_ch1_03', 'R_209_ch1_01', 'R_209_ch1_03', 'R_210_ch1_01', 'R_210_ch2_04', 'R_301_ch1_01',
    'R_301_ch1_04', 'R_302_ch1_01', 'R_302_ch1_04', 'R_303_ch1_01', 'R_303_ch1_04', 'R_304_ch1_01', 'R_304_ch1_03', 'R_305_ch1_01', 'R_305_ch1_04', 'R_313_ch1_01', 'R_313_ch1_03']

# 134ea, 60case # /NAS/DATA2/Video/Robot/Dataset2_60case # /data2/Video/Robot/Dataset2_60case
OOB_robot_60 = [
    'R_310_ch1_01', 'R_310_ch1_03', 'R_311_ch1_01', 'R_311_ch1_03', 'R_312_ch1_02', 'R_312_ch1_03', 'R_320_ch1_01', 'R_320_ch1_03', 'R_321_ch1_01', 'R_321_ch1_03', 
    'R_321_ch1_05', 'R_324_ch1_01', 'R_324_ch1_03', 'R_329_ch1_01', 'R_329_ch1_03', 'R_334_ch1_01', 'R_334_ch1_03', 'R_336_ch1_01', 'R_336_ch1_04', 'R_338_ch1_01', 
    'R_338_ch1_03', 'R_338_ch1_05', 'R_339_ch1_01', 'R_339_ch1_03', 'R_339_ch1_05', 'R_340_ch1_01', 'R_340_ch1_03', 'R_340_ch1_05', 'R_342_ch1_01', 'R_342_ch1_03', 
    'R_342_ch1_05', 'R_345_ch1_01', 'R_345_ch1_04', 'R_346_ch1_02', 'R_346_ch1_04', 'R_347_ch1_02', 'R_347_ch1_03', 'R_347_ch1_05', 'R_348_ch1_01', 'R_348_ch1_03', 
    'R_349_ch1_01', 'R_349_ch1_04', 'R_355_ch1_02', 'R_355_ch1_04', 'R_357_ch1_01', 'R_357_ch1_03', 'R_357_ch1_05', 'R_358_ch1_01', 'R_358_ch1_03', 'R_358_ch1_05', 
    'R_362_ch1_01', 'R_362_ch1_03', 'R_362_ch1_05', 'R_363_ch1_01', 'R_363_ch1_03', 'R_369_ch1_01', 'R_369_ch1_03', 'R_372_ch1_01', 'R_372_ch1_04', 'R_376_ch1_01', 
    'R_376_ch1_03', 'R_376_ch1_05', 'R_378_ch1_01', 'R_378_ch1_03', 'R_378_ch1_05', 'R_379_ch1_02', 'R_379_ch1_04', 'R_386_ch1_01', 'R_386_ch1_03', 'R_391_ch1_01', 
    'R_391_ch1_03', 'R_391_ch2_06', 'R_393_ch1_01', 'R_393_ch1_04', 'R_399_ch1_01', 'R_399_ch1_04', 'R_400_ch1_01', 'R_400_ch1_03', 'R_402_ch1_01', 'R_402_ch1_03', 
    'R_403_ch1_01', 'R_403_ch1_03', 'R_405_ch1_01', 'R_405_ch1_03', 'R_405_ch1_05', 'R_406_ch1_02', 'R_406_ch1_04', 'R_406_ch1_06', 'R_409_ch1_01', 'R_409_ch1_03', 
    'R_412_ch1_01', 'R_412_ch1_03', 'R_413_ch1_02', 'R_413_ch1_04', 'R_415_ch1_01', 'R_415_ch1_03', 'R_415_ch1_05', 'R_418_ch1_02', 'R_418_ch1_04', 'R_418_ch1_06', 
    'R_419_ch1_01', 'R_419_ch1_04', 'R_420_ch1_01', 'R_420_ch1_03', 'R_423_ch1_01', 'R_423_ch1_03', 'R_424_ch2_02', 'R_424_ch2_04', 'R_427_ch1_01', 'R_427_ch1_03', 
    'R_436_ch1_02', 'R_436_ch1_04', 'R_436_ch1_06', 'R_436_ch1_08', 'R_436_ch1_10', 'R_445_ch1_01', 'R_445_ch1_03', 'R_449_ch1_01', 'R_449_ch1_04', 'R_449_ch1_06', 
    'R_455_ch1_01', 'R_455_ch1_03', 'R_455_ch1_05', 'R_480_ch1_01', 'R_493_ch1_01', 'R_493_ch1_03', 'R_501_ch1_01', 'R_510_ch1_01', 'R_510_ch1_03', 'R_522_ch1_01', 
    'R_523_ch1_01', 'R_526_ch1_01', 'R_532_ch1_01', 'R_533_ch1_01']

# 225ea & 100case 
OOB_robot_list = OOB_robot_40 + OOB_robot_60

# 350ea, 40case # /NAS/DATA2/Public/IDC_21.06.25/Dataset1 # /data2/Public/IDC_21.06.25/Dataset1
OOB_lapa_40 = [
    'L_301_xx0_01', 'L_301_xx0_02', 'L_301_xx0_03', 'L_301_xx0_04', 'L_301_xx0_05', 'L_301_xx0_06', 'L_303_xx0_01', 'L_303_xx0_02', 'L_303_xx0_03', 'L_303_xx0_04', 
    'L_303_xx0_05', 'L_303_xx0_06', 'L_305_xx0_01', 'L_305_xx0_02', 'L_305_xx0_03', 'L_305_xx0_04', 'L_305_xx0_05', 'L_305_xx0_06', 'L_305_xx0_07', 'L_305_xx0_08', 
    'L_305_xx0_09', 'L_305_xx0_10', 'L_305_xx0_11', 'L_305_xx0_12', 'L_305_xx0_13', 'L_305_xx0_14', 'L_305_xx0_15', 'L_309_xx0_01', 'L_309_xx0_02', 'L_309_xx0_03', 
    'L_309_xx0_04', 'L_309_xx0_05', 'L_309_xx0_06', 'L_309_xx0_07', 'L_317_xx0_01', 'L_317_xx0_02', 'L_317_xx0_03', 'L_317_xx0_04', 'L_325_xx0_01', 'L_325_xx0_02', 
    'L_325_xx0_03', 'L_325_xx0_04', 'L_325_xx0_05', 'L_325_xx0_06', 'L_325_xx0_07', 'L_325_xx0_08', 'L_325_xx0_09', 'L_325_xx0_10', 'L_325_xx0_11', 'L_325_xx0_12', 
    'L_326_xx0_01', 'L_326_xx0_02', 'L_326_xx0_03', 'L_326_xx0_04', 'L_326_xx0_05', 'L_326_xx0_06', 'L_340_xx0_01', 'L_340_xx0_02', 'L_340_xx0_03', 'L_340_xx0_04', 
    'L_340_xx0_05', 'L_340_xx0_06', 'L_340_xx0_07', 'L_340_xx0_08', 'L_340_xx0_09', 'L_340_xx0_10', 'L_346_xx0_01', 'L_346_xx0_02', 'L_349_ch1_01', 'L_349_ch1_02', 
    'L_349_ch1_03', 'L_349_ch1_04', 'L_412_xx0_01', 'L_412_xx0_02', 'L_412_xx0_03', 'L_421_xx0_01', 'L_421_xx0_02', 'L_423_xx0_01', 'L_423_xx0_02', 'L_423_xx0_03', 
    'L_423_xx0_04', 'L_423_xx0_05', 'L_442_xx0_01', 'L_442_xx0_02', 'L_442_xx0_03', 'L_442_xx0_04', 'L_442_xx0_05', 'L_442_xx0_06', 'L_442_xx0_07', 'L_442_xx0_08', 
    'L_442_xx0_09', 'L_442_xx0_10', 'L_442_xx0_11', 'L_442_xx0_12', 'L_442_xx0_13', 'L_442_xx0_14', 'L_443_xx0_01', 'L_443_xx0_02', 'L_443_xx0_03', 'L_443_xx0_04', 
    'L_443_xx0_05', 'L_443_xx0_06', 'L_443_xx0_07', 'L_443_xx0_08', 'L_443_xx0_09', 'L_443_xx0_10', 'L_443_xx0_11', 'L_443_xx0_12', 'L_443_xx0_13', 'L_443_xx0_14', 
    'L_443_xx0_15', 'L_443_xx0_16', 'L_450_xx0_01', 'L_450_xx0_02', 'L_450_xx0_03', 'L_450_xx0_04', 'L_450_xx0_05', 'L_450_xx0_06', 'L_450_xx0_07', 'L_450_xx0_08', 
    'L_450_xx0_09', 'L_450_xx0_10', 'L_450_xx0_11', 'L_450_xx0_12', 'L_450_xx0_13', 'L_450_xx0_14', 'L_450_xx0_15', 'L_450_xx0_16', 'L_450_xx0_17', 'L_450_xx0_18', 
    'L_450_xx0_19', 'L_450_xx0_20', 'L_450_xx0_21', 'L_450_xx0_22', 'L_458_xx0_01', 'L_458_xx0_02', 'L_458_xx0_03', 'L_458_xx0_04', 'L_458_xx0_05', 'L_458_xx0_06', 
    'L_458_xx0_07', 'L_458_xx0_08', 'L_458_xx0_09', 'L_458_xx0_10', 'L_458_xx0_11', 'L_458_xx0_12', 'L_458_xx0_13', 'L_458_xx0_14', 'L_458_xx0_15', 'L_465_xx0_01', 
    'L_465_xx0_02', 'L_465_xx0_03', 'L_465_xx0_04', 'L_465_xx0_05', 'L_465_xx0_06', 'L_465_xx0_07', 'L_465_xx0_08', 'L_465_xx0_09', 'L_465_xx0_10', 'L_465_xx0_11', 
    'L_465_xx0_12', 'L_465_xx0_13', 'L_465_xx0_14', 'L_465_xx0_15', 'L_465_xx0_16', 'L_465_xx0_17', 'L_465_xx0_18', 'L_465_xx0_19', 'L_465_xx0_20', 'L_465_xx0_21', 
    'L_491_xx0_01', 'L_491_xx0_02', 'L_491_xx0_03', 'L_491_xx0_04', 'L_491_xx0_05', 'L_491_xx0_06', 'L_491_xx0_07', 'L_491_xx0_08', 'L_491_xx0_09', 'L_491_xx0_10', 
    'L_491_xx0_11', 'L_491_xx0_12', 'L_493_ch1_01', 'L_493_ch1_02', 'L_493_ch1_03', 'L_493_ch1_04', 'L_496_ch1_01', 'L_496_ch1_02', 'L_496_ch1_03', 'L_507_xx0_01', 
    'L_507_xx0_02', 'L_507_xx0_03', 'L_507_xx0_04', 'L_507_xx0_05', 'L_507_xx0_06', 'L_507_xx0_07', 'L_522_xx0_01', 'L_522_xx0_02', 'L_522_xx0_03', 'L_522_xx0_04', 
    'L_522_xx0_05', 'L_522_xx0_06', 'L_522_xx0_07', 'L_522_xx0_08', 'L_522_xx0_09', 'L_522_xx0_10', 'L_522_xx0_11', 'L_534_xx0_01', 'L_534_xx0_02', 'L_534_xx0_03', 
    'L_534_xx0_04', 'L_534_xx0_05', 'L_534_xx0_06', 'L_534_xx0_07', 'L_535_xx0_01', 'L_535_xx0_02', 'L_535_xx0_03', 'L_535_xx0_04', 'L_535_xx0_05', 'L_550_xx0_01', 
    'L_550_xx0_02', 'L_550_xx0_03', 'L_550_xx0_04', 'L_550_xx0_05', 'L_550_xx0_06', 'L_550_xx0_07', 'L_550_xx0_08', 'L_550_xx0_09', 'L_550_xx0_10', 'L_550_xx0_11', 
    'L_550_xx0_12', 'L_553_ch1_01', 'L_553_ch1_02', 'L_553_ch1_03', 'L_553_ch1_04', 'L_586_xx0_01', 'L_586_xx0_02', 'L_586_xx0_03', 'L_586_xx0_04', 'L_586_xx0_05', 
    'L_586_xx0_06', 'L_586_xx0_07', 'L_586_xx0_08', 'L_586_xx0_09', 'L_586_xx0_10', 'L_586_xx0_11', 'L_586_xx0_12', 'L_586_xx0_13', 'L_586_xx0_14', 'L_586_xx0_15', 
    'L_586_xx0_16', 'L_586_xx0_17', 'L_586_xx0_18', 'L_586_xx0_19', 'L_586_xx0_20', 'L_595_xx0_01', 'L_595_xx0_02', 'L_595_xx0_03', 'L_595_xx0_04', 'L_595_xx0_05', 
    'L_595_xx0_06', 'L_595_xx0_07', 'L_595_xx0_08', 'L_605_xx0_01', 'L_605_xx0_02', 'L_605_xx0_03', 'L_605_xx0_04', 'L_605_xx0_05', 'L_605_xx0_06', 'L_605_xx0_07', 
    'L_605_xx0_08', 'L_605_xx0_09', 'L_605_xx0_10', 'L_605_xx0_11', 'L_605_xx0_12', 'L_605_xx0_13', 'L_605_xx0_14', 'L_605_xx0_15', 'L_605_xx0_16', 'L_605_xx0_17', 
    'L_605_xx0_18', 'L_607_xx0_01', 'L_607_xx0_02', 'L_607_xx0_03', 'L_607_xx0_04', 'L_625_xx0_01', 'L_625_xx0_02', 'L_625_xx0_03', 'L_625_xx0_04', 'L_625_xx0_05', 
    'L_625_xx0_06', 'L_625_xx0_07', 'L_625_xx0_08', 'L_625_xx0_09', 'L_631_xx0_01', 'L_631_xx0_02', 'L_631_xx0_03', 'L_631_xx0_04', 'L_631_xx0_05', 'L_631_xx0_06', 
    'L_631_xx0_07', 'L_631_xx0_08', 'L_647_xx0_01', 'L_647_xx0_02', 'L_647_xx0_03', 'L_647_xx0_04', 'L_654_xx0_01', 'L_654_xx0_02', 'L_654_xx0_03', 'L_654_xx0_04', 
    'L_654_xx0_05', 'L_654_xx0_06', 'L_654_xx0_07', 'L_654_xx0_08', 'L_654_xx0_09', 'L_654_xx0_10', 'L_654_xx0_11', 'L_659_xx0_01', 'L_659_xx0_02', 'L_659_xx0_03', 
    'L_659_xx0_04', 'L_659_xx0_05', 'L_660_xx0_01', 'L_660_xx0_02', 'L_660_xx0_03', 'L_660_xx0_04', 'L_661_xx0_01', 'L_661_xx0_02', 'L_661_xx0_03', 'L_661_xx0_04', 
    'L_661_xx0_05', 'L_661_xx0_06', 'L_661_xx0_07', 'L_661_xx0_08', 'L_661_xx0_09', 'L_661_xx0_10', 'L_661_xx0_11', 'L_661_xx0_12', 'L_661_xx0_13', 'L_661_xx0_14', 
    'L_661_xx0_15', 'L_669_xx0_01', 'L_669_xx0_02', 'L_669_xx0_03', 'L_669_xx0_04', 'L_676_xx0_01', 'L_676_xx0_02', 'L_676_xx0_03', 'L_676_xx0_04', 'L_676_xx0_05']

# 521ea, 60case # /NAS/DATA2/Public/IDC_21.06.25/Dataset2 # /data2/Public/IDC_21.06.25/Dataset2
OOB_lapa_60 = [
    'L_310_xx0_01', 'L_310_xx0_02', 'L_310_xx0_03', 'L_310_xx0_04', 'L_310_xx0_05', 'L_310_xx0_06', 'L_310_xx0_07', 'L_310_xx0_08', 'L_310_xx0_09', 'L_310_xx0_10', 
    'L_310_xx0_11', 'L_310_xx0_12', 'L_311_xx0_01', 'L_311_xx0_02', 'L_311_xx0_03', 'L_311_xx0_04', 'L_311_xx0_05', 'L_330_ch1_01', 'L_333_xx0_01', 'L_333_xx0_02', 
    'L_333_xx0_03', 'L_333_xx0_04', 'L_333_xx0_05', 'L_333_xx0_06', 'L_333_xx0_07', 'L_333_xx0_08', 'L_333_xx0_09', 'L_333_xx0_10', 'L_333_xx0_11', 'L_367_ch1_01', 
    'L_370_ch1_01', 'L_377_ch1_01', 'L_379_xx0_01', 'L_379_xx0_02', 'L_379_xx0_03', 'L_379_xx0_04', 'L_379_xx0_05', 'L_379_xx0_06', 'L_379_xx0_07', 'L_379_xx0_08', 
    'L_379_xx0_09', 'L_379_xx0_10', 'L_379_xx0_11', 'L_385_xx0_01', 'L_385_xx0_02', 'L_385_xx0_03', 'L_385_xx0_04', 'L_385_xx0_05', 'L_385_xx0_06', 'L_385_xx0_07', 
    'L_385_xx0_08', 'L_385_xx0_09', 'L_385_xx0_10', 'L_385_xx0_11', 'L_385_xx0_12', 'L_385_xx0_13', 'L_385_xx0_14', 'L_385_xx0_15', 'L_387_xx0_01', 'L_387_xx0_02', 
    'L_387_xx0_03', 'L_387_xx0_04', 'L_387_xx0_05', 'L_387_xx0_06', 'L_387_xx0_07', 'L_387_xx0_08', 'L_389_xx0_01', 'L_389_xx0_02', 'L_389_xx0_03', 'L_389_xx0_04', 
    'L_389_xx0_05', 'L_389_xx0_06', 'L_389_xx0_07', 'L_389_xx0_08', 'L_389_xx0_09', 'L_389_xx0_10', 'L_389_xx0_11', 'L_389_xx0_12', 'L_389_xx0_13', 'L_391_xx0_01', 
    'L_391_xx0_02', 'L_391_xx0_03', 'L_391_xx0_04', 'L_391_xx0_05', 'L_391_xx0_06', 'L_391_xx0_07', 'L_391_xx0_08', 'L_391_xx0_09', 'L_393_xx0_01', 'L_393_xx0_02', 
    'L_393_xx0_03', 'L_393_xx0_04', 'L_393_xx0_05', 'L_393_xx0_06', 'L_393_xx0_07', 'L_393_xx0_08', 'L_393_xx0_09', 'L_393_xx0_10', 'L_400_xx0_01', 'L_400_xx0_02', 
    'L_400_xx0_03', 'L_400_xx0_04', 'L_400_xx0_05', 'L_400_xx0_06', 'L_400_xx0_07', 'L_400_xx0_08', 'L_400_xx0_09', 'L_400_xx0_10', 'L_400_xx0_11', 'L_400_xx0_12', 
    'L_402_xx0_01', 'L_402_xx0_02', 'L_402_xx0_03', 'L_402_xx0_04', 'L_406_xx0_01', 'L_406_xx0_02', 'L_406_xx0_03', 'L_406_xx0_04', 'L_406_xx0_05', 'L_406_xx0_06', 
    'L_406_xx0_07', 'L_406_xx0_08', 'L_406_xx0_09', 'L_406_xx0_10', 'L_406_xx0_11', 'L_406_xx0_12', 'L_406_xx0_13', 'L_408_ch1_01', 'L_413_xx0_01', 'L_413_xx0_02', 
    'L_413_xx0_03', 'L_413_xx0_04', 'L_413_xx0_05', 'L_413_xx0_06', 'L_413_xx0_07', 'L_413_xx0_08', 'L_413_xx0_09', 'L_413_xx0_10', 'L_414_xx0_01', 'L_414_xx0_02', 
    'L_414_xx0_03', 'L_414_xx0_04', 'L_414_xx0_05', 'L_414_xx0_06', 'L_414_xx0_07', 'L_414_xx0_08', 'L_415_xx0_01', 'L_415_xx0_02', 'L_415_xx0_03', 'L_415_xx0_04', 
    'L_415_xx0_05', 'L_415_xx0_06', 'L_415_xx0_07', 'L_415_xx0_08', 'L_415_xx0_09', 'L_415_xx0_10', 'L_415_xx0_11', 'L_415_xx0_12', 'L_418_xx0_01', 'L_418_xx0_02', 
    'L_418_xx0_03', 'L_418_xx0_04', 'L_418_xx0_05', 'L_418_xx0_06', 'L_418_xx0_07', 'L_418_xx0_08', 'L_419_xx0_01', 'L_419_xx0_02', 'L_419_xx0_03', 'L_419_xx0_04', 
    'L_419_xx0_05', 'L_419_xx0_06', 'L_427_xx0_01', 'L_427_xx0_02', 'L_427_xx0_03', 'L_427_xx0_04', 'L_427_xx0_05', 'L_427_xx0_06', 'L_427_xx0_07', 'L_427_xx0_08', 
    'L_427_xx0_09', 'L_427_xx0_10', 'L_427_xx0_11', 'L_427_xx0_12', 'L_427_xx0_13', 'L_427_xx0_14', 'L_427_xx0_15', 'L_428_ch1_01', 'L_430_ch1_01', 'L_433_xx0_01', 
    'L_433_xx0_02', 'L_433_xx0_03', 'L_433_xx0_04', 'L_433_xx0_05', 'L_433_xx0_06', 'L_433_xx0_07', 'L_433_xx0_08', 'L_433_xx0_09', 'L_434_xx0_01', 'L_434_xx0_02', 
    'L_434_xx0_03', 'L_434_xx0_04', 'L_434_xx0_05', 'L_434_xx0_06', 'L_434_xx0_07', 'L_434_xx0_08', 'L_434_xx0_09', 'L_434_xx0_10', 'L_436_xx0_01', 'L_436_xx0_02', 
    'L_436_xx0_03', 'L_436_xx0_04', 'L_436_xx0_05', 'L_436_xx0_06', 'L_436_xx0_07', 'L_436_xx0_08', 'L_436_xx0_09', 'L_436_xx0_10', 'L_436_xx0_11', 'L_436_xx0_12', 
    'L_439_xx0_01', 'L_439_xx0_02', 'L_439_xx0_03', 'L_439_xx0_04', 'L_439_xx0_05', 'L_439_xx0_06', 'L_439_xx0_07', 'L_439_xx0_08', 'L_439_xx0_09', 'L_439_xx0_10', 
    'L_439_xx0_11', 'L_439_xx0_12', 'L_439_xx0_13', 'L_439_xx0_14', 'L_439_xx0_15', 'L_439_xx0_16', 'L_471_xx0_01', 'L_471_xx0_02', 'L_471_xx0_03', 'L_471_xx0_04', 
    'L_471_xx0_05', 'L_471_xx0_06', 'L_471_xx0_07', 'L_471_xx0_08', 'L_471_xx0_09', 'L_471_xx0_10', 'L_471_xx0_11', 'L_473_xx0_01', 'L_473_xx0_02', 'L_473_xx0_03', 
    'L_473_xx0_04', 'L_473_xx0_05', 'L_473_xx0_06', 'L_473_xx0_07', 'L_475_ch1_01', 'L_475_ch1_02', 'L_477_ch1_01', 'L_478_xx0_01', 'L_478_xx0_02', 'L_478_xx0_03', 
    'L_478_xx0_04', 'L_478_xx0_05', 'L_478_xx0_06', 'L_478_xx0_07', 'L_478_xx0_08', 'L_478_xx0_09', 'L_478_xx0_10', 'L_479_xx0_01', 'L_479_xx0_02', 'L_479_xx0_03', 
    'L_479_xx0_04', 'L_479_xx0_05', 'L_479_xx0_06', 'L_479_xx0_07', 'L_479_xx0_08', 'L_479_xx0_09', 'L_481_xx0_01', 'L_481_xx0_02', 'L_481_xx0_03', 'L_481_xx0_04', 
    'L_481_xx0_05', 'L_481_xx0_06', 'L_481_xx0_07', 'L_481_xx0_08', 'L_481_xx0_09', 'L_481_xx0_10', 'L_481_xx0_11', 'L_481_xx0_12', 'L_481_xx0_13', 'L_482_xx0_01', 
    'L_482_xx0_02', 'L_482_xx0_03', 'L_482_xx0_04', 'L_482_xx0_05', 'L_482_xx0_06', 'L_482_xx0_07', 'L_482_xx0_08', 'L_482_xx0_09', 'L_482_xx0_10', 'L_482_xx0_11', 
    'L_482_xx0_12', 'L_482_xx0_13', 'L_482_xx0_14', 'L_482_xx0_15', 'L_484_xx0_01', 'L_484_xx0_02', 'L_484_xx0_03', 'L_484_xx0_04', 'L_484_xx0_05', 'L_484_xx0_06', 
    'L_484_xx0_07', 'L_484_xx0_08', 'L_484_xx0_09', 'L_484_xx0_10', 'L_484_xx0_11', 'L_513_xx0_01', 'L_513_xx0_02', 'L_513_xx0_03', 'L_513_xx0_04', 'L_513_xx0_05', 
    'L_513_xx0_06', 'L_513_xx0_07', 'L_513_xx0_08', 'L_513_xx0_09', 'L_513_xx0_10', 'L_513_xx0_11', 'L_513_xx0_12', 'L_514_xx0_01', 'L_514_xx0_02', 'L_514_xx0_03', 
    'L_514_xx0_04', 'L_514_xx0_05', 'L_514_xx0_06', 'L_514_xx0_07', 'L_514_xx0_08', 'L_514_xx0_09', 'L_514_xx0_10', 'L_515_xx0_01', 'L_515_xx0_02', 'L_515_xx0_03', 
    'L_515_xx0_04', 'L_515_xx0_05', 'L_515_xx0_06', 'L_515_xx0_07', 'L_515_xx0_08', 'L_517_xx0_01', 'L_517_xx0_02', 'L_517_xx0_03', 'L_517_xx0_04', 'L_517_xx0_05', 
    'L_517_xx0_06', 'L_517_xx0_07', 'L_517_xx0_08', 'L_537_xx0_01', 'L_537_xx0_02', 'L_537_xx0_03', 'L_537_xx0_04', 'L_537_xx0_05', 'L_537_xx0_06', 'L_537_xx0_07', 
    'L_537_xx0_08', 'L_537_xx0_09', 'L_537_xx0_10', 'L_537_xx0_11', 'L_537_xx0_12', 'L_539_ch1_01', 'L_542_xx0_01', 'L_542_xx0_02', 'L_542_xx0_03', 'L_542_xx0_04', 
    'L_542_xx0_05', 'L_542_xx0_06', 'L_542_xx0_07', 'L_542_xx0_08', 'L_542_xx0_09', 'L_543_xx0_01', 'L_543_xx0_02', 'L_543_xx0_03', 'L_543_xx0_04', 'L_543_xx0_05', 
    'L_543_xx0_06', 'L_543_xx0_07', 'L_543_xx0_08', 'L_543_xx0_09', 'L_543_xx0_10', 'L_543_xx0_11', 'L_543_xx0_12', 'L_543_xx0_13', 'L_543_xx0_14', 'L_545_xx0_01', 
    'L_545_xx0_02', 'L_545_xx0_03', 'L_545_xx0_04', 'L_545_xx0_05', 'L_545_xx0_06', 'L_545_xx0_07', 'L_545_xx0_08', 'L_545_xx0_09', 'L_546_xx0_01', 'L_546_xx0_02', 
    'L_546_xx0_03', 'L_546_xx0_04', 'L_546_xx0_05', 'L_546_xx0_06', 'L_546_xx0_07', 'L_546_xx0_08', 'L_546_xx0_09', 'L_546_xx0_10', 'L_556_xx0_01', 'L_556_xx0_02', 
    'L_556_xx0_03', 'L_556_xx0_04', 'L_556_xx0_05', 'L_556_xx0_06', 'L_556_xx0_07', 'L_556_xx0_08', 'L_556_xx0_09', 'L_556_xx0_10', 'L_556_xx0_11', 'L_556_xx0_12', 
    'L_556_xx0_13', 'L_556_xx0_14', 'L_558_xx0_01', 'L_560_xx0_01', 'L_560_xx0_02', 'L_560_xx0_03', 'L_560_xx0_04', 'L_560_xx0_05', 'L_560_xx0_06', 'L_560_xx0_07', 
    'L_560_xx0_08', 'L_560_xx0_09', 'L_560_xx0_10', 'L_560_xx0_11', 'L_560_xx0_12', 'L_560_xx0_13', 'L_560_xx0_14', 'L_560_xx0_15', 'L_563_xx0_01', 'L_563_xx0_02', 
    'L_563_xx0_03', 'L_563_xx0_04', 'L_563_xx0_05', 'L_563_xx0_06', 'L_563_xx0_07', 'L_563_xx0_08', 'L_563_xx0_09', 'L_563_xx0_10', 'L_563_xx0_11', 'L_563_xx0_12', 
    'L_565_xx0_01', 'L_565_xx0_02', 'L_565_xx0_03', 'L_565_xx0_04', 'L_565_xx0_05', 'L_565_xx0_06', 'L_565_xx0_07', 'L_565_xx0_08', 'L_565_xx0_09', 'L_568_xx0_01', 
    'L_568_xx0_02', 'L_568_xx0_03', 'L_568_xx0_04', 'L_568_xx0_05', 'L_568_xx0_06', 'L_568_xx0_07', 'L_568_xx0_08', 'L_568_xx0_09', 'L_569_xx0_01', 'L_569_xx0_02', 
    'L_569_xx0_03', 'L_569_xx0_04', 'L_569_xx0_05', 'L_569_xx0_06', 'L_569_xx0_07', 'L_569_xx0_08', 'L_569_xx0_09', 'L_569_xx0_10', 'L_569_xx0_11', 'L_569_xx0_12', 
    'L_572_ch1_01', 'L_574_xx0_01', 'L_574_xx0_02', 'L_574_xx0_03', 'L_574_xx0_04', 'L_574_xx0_05', 'L_574_xx0_06', 'L_574_xx0_07', 'L_574_xx0_08', 'L_574_xx0_09', 
    'L_574_xx0_10', 'L_574_xx0_11', 'L_575_xx0_01', 'L_575_xx0_02', 'L_575_xx0_03', 'L_575_xx0_04', 'L_577_xx0_01', 'L_577_xx0_02', 'L_577_xx0_03', 'L_577_xx0_04', 
    'L_577_xx0_05', 'L_577_xx0_06', 'L_577_xx0_07', 'L_577_xx0_08', 'L_577_xx0_09', 'L_577_xx0_10', 'L_580_xx0_01', 'L_580_xx0_02', 'L_580_xx0_03', 'L_580_xx0_04', 
    'L_580_xx0_05', 'L_580_xx0_06', 'L_580_xx0_07', 'L_580_xx0_08', 'L_580_xx0_09', 'L_580_xx0_10', 'L_580_xx0_11', 'L_580_xx0_12', 'L_580_xx0_13', 'L_580_xx0_14', 'L_580_xx0_15']

# 871ea, 100case
OOB_lapa_list = OOB_lapa_40 + OOB_lapa_60

def visual_oob_event(anno_meta_info_csv_path):
    # columns=[Patient, 'Method', totalFrame, fps, IB_count, OOB_count, total_time, IB_event_time, OOB_event_time, OOB_event_cnt, annotation_start_point, annotation_end_point, start_frame_idx,end_frame_idx, OOB_event_duration, start_frame_time, end_frame_time]
    anno_meta_info_df = pd.read_csv(anno_meta_info_csv_path, index_col=None)

    '''
    # append method [R, L]
    method_list = anno_meta_info_df['Patient'].str.split('_')
    anno_meta_info_df['method'] = method_list.str.get(0)
    print(anno_meta_info_df)
    '''
    
    # check oob duration
    '''
    min_event_duration = 60.0
    max_event_duration = 100000000000.0
    over_min_event_duration = anno_meta_info_df['OOB_event_duration'] > min_event_duration
    under_max_event_duration = anno_meta_info_df['OOB_event_duration'] < max_event_duration
    

    # extract target oob duration df
    anno_meta_info_df = anno_meta_info_df[over_min_event_duration & under_max_event_duration]
    '''
    
    
    # normalize
    anno_meta_info_df['start_frame_idx_normalized'] = anno_meta_info_df['start_frame_idx'] / anno_meta_info_df['totalFrame']
    anno_meta_info_df['end_frame_idx_normalized'] = anno_meta_info_df['end_frame_idx'] / anno_meta_info_df['totalFrame']
    

    anno_meta_info_df = anno_meta_info_df.sort_values(['OOB_event_duration'], ascending=[True])
    print(anno_meta_info_df)
    
    # unique info df
    unique_anno_meta_info_df = anno_meta_info_df[['Patient', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'IB_event_time', 'OOB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point']].drop_duplicates()
    print(unique_anno_meta_info_df)

    # print(len(anno_meta_info_df[anno_meta_info_df['Method']=='R'])) # 1247
    # print(len(anno_meta_info_df[anno_meta_info_df['Method']=='L'])) # 958

    
    # pairplot

    anno_event_cnt_info_df = unique_anno_meta_info_df[['Method', 'OOB_event_cnt', 'OOB_event_time']]

    oob_event_cnt_sns_pairplot = sns.pairplot(data=anno_event_cnt_info_df, hue='Method', kind='reg')
    oob_event_cnt_sns_pairplot.fig.suptitle('Relationship of OOB EVENT CNT & OOB_EVENT_TIME (total)')

    
    oob_event_cnt_sns_pairplot.fig.subplots_adjust(top=0.9)
    
    oob_event_cnt_sns_pairplot.savefig('./OOB_EVENT_VISUAL/LAPA_OOB_EVENT_DURATION_per_CNT_TOTAL.png')
    

    '''
    # jointplot
    oob_event_time_sns_jointplot = sns.jointplot(data=anno_meta_info_df, x='start_frame_time', y='end_frame_time', hue='Method', alpha=0.5) # kind='reg'
    oob_event_time_sns_jointplot.fig.suptitle('OOB EVENT TIME Distridution')
    oob_event_time_sns_jointplot.fig.tight_layout()
    oob_event_time_sns_jointplot.fig.subplots_adjust(top=0.95)
    
    oob_event_time_sns_jointplot.savefig('./OOB_EVENT_VISUAL/OOB_EVENT_TIME_DISTRIBUTION_TOTAL.png')
    '''
    
    
    # jointplot
    
    oob_event_time_sns_jointplot = sns.jointplot(data=anno_meta_info_df, x='start_frame_idx_normalized', y='end_frame_idx_normalized', kind='hex', marginal_ticks=True, xlim=(0, 1.2), ylim=(0, 1.2), color='orange') # kind='reg'
    oob_event_time_sns_jointplot.fig.suptitle('LAPA OOB EVENT TIME Distridution (total)')
    oob_event_time_sns_jointplot.fig.tight_layout()
    oob_event_time_sns_jointplot.fig.subplots_adjust(top=0.95)
    
    oob_event_time_sns_jointplot.savefig('./OOB_EVENT_VISUAL/LAPA_OOB_EVENT_TIME_DISTRIBUTION_TOTAL.png')
    
    
def visual_oob_event_time(anno_meta_info_csv_path):
    # columns=[Patient, 'Method', totalFrame, fps, IB_count, OOB_count, total_time, IB_event_time, OOB_event_time, OOB_event_cnt, annotation_start_point, annotation_end_point, start_frame_idx,end_frame_idx, OOB_event_duration, start_frame_time, end_frame_time]
    anno_meta_info_df = pd.read_csv(anno_meta_info_csv_path, index_col=None)

    '''
    # append method [R, L]
    method_list = anno_meta_info_df['Patient'].str.split('_')
    anno_meta_info_df['method'] = method_list.str.get(0)
    print(anno_meta_info_df)
    '''
    
    # check oob duration
    '''
    min_event_duration = 60.0
    max_event_duration = 100000000000.0
    over_min_event_duration = anno_meta_info_df['OOB_event_duration'] > min_event_duration
    under_max_event_duration = anno_meta_info_df['OOB_event_duration'] < max_event_duration
    

    # extract target oob duration df
    anno_meta_info_df = anno_meta_info_df[over_min_event_duration & under_max_event_duration]
    '''
    
    
    # normalize
    anno_meta_info_df['OOB_event_time_normalized'] = (anno_meta_info_df['OOB_event_time'] / anno_meta_info_df['total_time']) * 100
    anno_meta_info_df['IB_event_time_normalized'] = (anno_meta_info_df['IB_event_time'] / anno_meta_info_df['total_time']) * 100
    

    anno_meta_info_df = anno_meta_info_df.sort_values(['OOB_event_time_normalized'], ascending=[False])
    print(anno_meta_info_df)
    
    # unique info df
    unique_anno_meta_info_df = anno_meta_info_df[['Patient', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'IB_event_time', 'OOB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'OOB_event_time_normalized', 'IB_event_time_normalized']].drop_duplicates()
    print(unique_anno_meta_info_df)

    # clac box plot info
    grouped = unique_anno_meta_info_df.groupby('Method')
    grouped_dict = dict(list(grouped))
    print(grouped['OOB_event_time'].describe())
    print(grouped['OOB_event_time_normalized'].describe())
    
    robot_df = grouped_dict['R']
    lapa_df = grouped_dict['L']

    describe_oob_event_time = grouped['OOB_event_time_normalized'].describe()
    print(describe_oob_event_time.loc['L', '50%'])
    print(describe_oob_event_time.loc['R', '50%'])

    # robot
    q1_robot = robot_df[['OOB_event_time_normalized']].quantile(0.25, interpolation='lower')[0]
    q3_robot = robot_df[['OOB_event_time_normalized']].quantile(0.75, interpolation='lower')[0]
    median_robot = robot_df[['OOB_event_time_normalized']].quantile(0.5, interpolation='lower')[0]
    min_robot = robot_df[['OOB_event_time_normalized']].quantile(0.0, interpolation='lower')[0]
    max_robot = robot_df[['OOB_event_time_normalized']].quantile(1.0, interpolation='lower')[0]
    iqr_robot = q3_robot - q1_robot

    fence_low_robot = q1_robot - (1.5 * iqr_robot)
    fence_high_robot = q3_robot + (1.5 * iqr_robot)

    q3_whisper_robot = robot_df[robot_df['OOB_event_time_normalized'] < fence_high_robot].sort_values('OOB_event_time_normalized', ascending=[True]).tail(1) # q3 whisper

    q1_robot_idx = robot_df.index[robot_df['OOB_event_time_normalized'] == q1_robot].tolist()
    q3_robot_idx = robot_df.index[robot_df['OOB_event_time_normalized'] == q3_robot].tolist()
    median_robot_idx = robot_df.index[robot_df['OOB_event_time_normalized'] == median_robot].tolist()
    max_robot_idx = robot_df.index[robot_df['OOB_event_time_normalized'] == max_robot].tolist()
    min_robot_idx = robot_df.index[robot_df['OOB_event_time_normalized'] == min_robot].tolist()


    print(q1_robot, q3_robot, iqr_robot, fence_low_robot, fence_high_robot, min_robot, max_robot)
    print(q1_robot_idx, q3_robot_idx, median_robot_idx, max_robot_idx, min_robot_idx)


    # lapa
    q1_lapa = lapa_df[['OOB_event_time_normalized']].quantile(0.25, interpolation='lower')[0]
    q3_lapa = lapa_df[['OOB_event_time_normalized']].quantile(0.75, interpolation='lower')[0]
    median_lapa = lapa_df[['OOB_event_time_normalized']].quantile(0.5, interpolation='lower')[0]
    min_lapa = lapa_df[['OOB_event_time_normalized']].quantile(0.0, interpolation='lower')[0]
    max_lapa = lapa_df[['OOB_event_time_normalized']].quantile(1.0, interpolation='lower')[0]
    iqr_lapa = q3_lapa - q1_lapa

    fence_low_lapa = q1_lapa - (1.5 * iqr_lapa)
    fence_high_lapa = q3_lapa + (1.5 * iqr_lapa)

    q3_whisper_lapa = lapa_df[lapa_df['OOB_event_time_normalized'] < fence_high_lapa].sort_values('OOB_event_time_normalized', ascending=[True]).tail(1) # q3 whisper

    q1_lapa_idx = lapa_df.index[lapa_df['OOB_event_time_normalized'] == q1_lapa].tolist()
    q3_lapa_idx = lapa_df.index[lapa_df['OOB_event_time_normalized'] == q3_lapa].tolist()
    median_lapa_idx = lapa_df.index[lapa_df['OOB_event_time_normalized'] == median_lapa].tolist()
    min_lapa_idx = lapa_df.index[lapa_df['OOB_event_time_normalized'] == min_lapa].tolist()
    max_lapa_idx = lapa_df.index[lapa_df['OOB_event_time_normalized'] == max_lapa].tolist()

    print(q1_lapa, q3_lapa, iqr_lapa, fence_low_lapa, fence_high_lapa, min_lapa, max_lapa)
    print(q1_lapa_idx, q3_lapa_idx, median_lapa_idx, min_lapa_idx, max_lapa_idx)

    print(lapa_df.loc[q1_lapa_idx, 'OOB_event_time'])

    # outlier
    outlier_robot = (robot_df['OOB_event_time_normalized'] <= fence_low_robot) | (robot_df['OOB_event_time_normalized'] >= fence_high_robot)
    outlier_lapa = (lapa_df['OOB_event_time_normalized'] <= fence_low_lapa) | (lapa_df['OOB_event_time_normalized'] >= fence_high_lapa)


    print(robot_df[outlier_robot])
    print(lapa_df[outlier_lapa])


    # boxplotting
    fig, ax = plt.subplots(figsize=(12,8), tight_layout=True)

    oob_event_time_sns_boxplot = sns.boxplot(y='OOB_event_time_normalized', x='Method', data=unique_anno_meta_info_df, palette='colorblind', hue='Method', ax=ax, showmeans=True)
    sns.stripplot(y='OOB_event_time_normalized', x='Method', data=unique_anno_meta_info_df, jitter=True, dodge=True, marker='o', alpha=0.3, hue='Method', color='grey')

    

    
    # ROBOT
    ax.text(0, describe_oob_event_time.loc['R', '25%'], '{} | {:.1f}s ({:.2f}%)'.format(robot_df.loc[q1_robot_idx[0], 'Patient'], robot_df.loc[q1_robot_idx[0], 'OOB_event_time'], robot_df.loc[q1_robot_idx[0], 'OOB_event_time_normalized']), horizontalalignment='left',  color='b', weight='semibold')
    ax.text(0, describe_oob_event_time.loc['R', '50%'], '{} |{:.1f}s ({:.2f}%)'.format(robot_df.loc[median_robot_idx[0], 'Patient'], robot_df.loc[median_robot_idx[0], 'OOB_event_time'], robot_df.loc[median_robot_idx[0], 'OOB_event_time_normalized']), horizontalalignment='left',  color='orange', weight='semibold')
    ax.text(0, describe_oob_event_time.loc['R', '75%'], '{} | {:.1f}s ({:.2f}%)'.format(robot_df.loc[q3_robot_idx[0], 'Patient'], robot_df.loc[q3_robot_idx[0], 'OOB_event_time'], robot_df.loc[q3_robot_idx[0], 'OOB_event_time_normalized']), horizontalalignment='left',  color='b', weight='semibold')
    ax.text(0, describe_oob_event_time.loc['R', 'mean'], '{:.2f}%'.format(describe_oob_event_time.loc['R', 'mean']), horizontalalignment='left',  color='r', weight='semibold')
    ax.text(0, describe_oob_event_time.loc['R', 'min'], '{} | {:.1f}s ({:.2f}%)'.format(robot_df.loc[min_robot_idx[0], 'Patient'], robot_df.loc[min_robot_idx[0], 'OOB_event_time'], robot_df.loc[min_robot_idx[0], 'OOB_event_time_normalized']), horizontalalignment='left',  color='g', weight='semibold')
    ax.text(0, describe_oob_event_time.loc['R', 'max'], '{} | {:.1f}s ({:.2f}%)'.format(robot_df.loc[max_robot_idx[0], 'Patient'], robot_df.loc[max_robot_idx[0], 'OOB_event_time'], robot_df.loc[max_robot_idx[0], 'OOB_event_time_normalized']), horizontalalignment='left',  color='g', weight='semibold')
    ax.text(0, q3_whisper_robot['OOB_event_time_normalized'].item(), '{} | {:.1f}s ({:.2f}%)'.format(q3_whisper_robot['Patient'].item(), q3_whisper_robot['OOB_event_time'].item(), q3_whisper_robot['OOB_event_time_normalized'].item()), horizontalalignment='left',  color='purple', weight='semibold')

    # LAPA
    ax.text(1, describe_oob_event_time.loc['L', '25%'], '{} | {:.1f}s ({:.2f}%)'.format(lapa_df.loc[q1_lapa_idx[0], 'Patient'], lapa_df.loc[q1_lapa_idx[0], 'OOB_event_time'], lapa_df.loc[q1_lapa_idx[0], 'OOB_event_time_normalized']), horizontalalignment='right',  color='b', weight='semibold')
    ax.text(1, describe_oob_event_time.loc['L', '50%'], '{} | {:.1f}s ({:.2f}%)'.format(lapa_df.loc[median_lapa_idx[0], 'Patient'], lapa_df.loc[median_lapa_idx[0], 'OOB_event_time'], lapa_df.loc[median_lapa_idx[0], 'OOB_event_time_normalized']), horizontalalignment='right',  color='orange', weight='semibold')
    ax.text(1, describe_oob_event_time.loc['L', '75%'], '{} | {:.1f}s ({:.2f})%'.format(lapa_df.loc[q3_lapa_idx[0], 'Patient'], lapa_df.loc[q3_lapa_idx[0], 'OOB_event_time'], lapa_df.loc[q3_lapa_idx[0], 'OOB_event_time_normalized']), horizontalalignment='right',  color='b', weight='semibold')
    ax.text(1, describe_oob_event_time.loc['L', 'mean'], '{:.2f}%'.format(describe_oob_event_time.loc['L', 'mean']), horizontalalignment='right',  color='r', weight='semibold')
    ax.text(1, describe_oob_event_time.loc['L', 'min'], '{} | {:.1f}s ({:.2f}%)'.format(lapa_df.loc[min_lapa_idx[0], 'Patient'], lapa_df.loc[min_lapa_idx[0], 'OOB_event_time'], lapa_df.loc[min_lapa_idx[0], 'OOB_event_time_normalized']), horizontalalignment='right',  color='g', weight='semibold')
    ax.text(1, describe_oob_event_time.loc['L', 'max'], '{} | {:.1f}s ({:.2f}%)'.format(lapa_df.loc[max_lapa_idx[0], 'Patient'], lapa_df.loc[max_lapa_idx[0], 'OOB_event_time'], lapa_df.loc[max_lapa_idx[0], 'OOB_event_time_normalized']), horizontalalignment='right',  color='g', weight='semibold')
    ax.text(1, q3_whisper_lapa['OOB_event_time_normalized'].item(), '{} | {:.1f}s ({:.2f}%)'.format(q3_whisper_lapa['Patient'].item(), q3_whisper_lapa['OOB_event_time'].item(), q3_whisper_lapa['OOB_event_time_normalized'].item()), horizontalalignment='right',  color='purple', weight='semibold')

    # outlier ploting - robot
    cnt = 0
    for idx in robot_df[outlier_robot].sort_values(['OOB_event_time_normalized'], ascending=[False]).index :
        cnt += 1
        
        if cnt in [1,6,7,8,9] :
            continue
        
        
        ax.text(0, robot_df.loc[idx, 'OOB_event_time_normalized'], '{} | {:.1f}s ({:.2f}%)'.format(robot_df.loc[idx, 'Patient'], robot_df.loc[idx, 'OOB_event_time'], robot_df.loc[idx, 'OOB_event_time_normalized']), horizontalalignment='left',  color='black')

    
    # outlier ploting - lapa
    cnt = 0
    for idx in lapa_df[outlier_lapa].sort_values(['OOB_event_time_normalized'], ascending=[False]).index :
        cnt += 1
        
        if cnt in [1,3,9,12] :
            continue
        
        
        ax.text(1, lapa_df.loc[idx, 'OOB_event_time_normalized'], '{} | {:.1f}s ({:.2f}%)'.format(lapa_df.loc[idx, 'Patient'], lapa_df.loc[idx, 'OOB_event_time'], lapa_df.loc[idx, 'OOB_event_time_normalized']), horizontalalignment='right',  color='black')

    # set title
    fig.suptitle('GASTRECTOMY OF OOB EVENT TIME (Normalized)')
    ax.grid()
    

    
    

    '''
    for label_idx, tick in enumerate(ax.get_xticklabels()):
        label = tick.get_text() # 'R', 'L'
        print(label_idx, label)
        ax.text(label_idx, describe_oob_event_time.loc[label, '25%'], '{:.3f} | {:.3f}'.format(,1), horizontalalignment='left',  color='b', weight='semibold')
    '''
        
    
    

    # print(len(anno_meta_info_df[anno_meta_info_df['Method']=='R'])) # 1247
    # print(len(anno_meta_info_df[anno_meta_info_df['Method']=='L'])) # 958

    
    # pairplot
    plt.savefig('./OOB_EVENT_VISUAL/GASTRECTOMY_OOB_EVENT_TIME(Normalized).png')
    



if __name__ == '__main__':
    # fold_inference(['fold1', 'fold2', 'fold3'])

    # new_data_path = './results_v2_robot_oob-mobilenet_v3_large-fold_1-last/Patient_Total_metric-ROBOT-results_v2_robot_oob-mobilenet_v3_large-fold_1-last.csv'
    # data_compare(new_data_path, output_path)

    visual_metric_per_patients_ver2('./results_v2_new_lapa_oob-mobilenet_v3_large_from_robot-fold_1-last-fps1_30/Patient_Total_metric-LAPA-results_v2_new_lapa_oob-mobilenet_v3_large_from_robot-fold_1-last-fps1_30.csv', 'mobilenet_v3_large', './results_v2_new_lapa_oob-mobilenet_v3_large_from_robot-fold_1-last-fps1_30/Patient_Total_metric-LAPA-results_v2_new_lapa_oob-mobilenet_v3_large_from_robot-fold_1-last-fps1_30.png')
    
    # anno_meta_info_csv_path = './DATA_SHEET/TOTAL_V2_anno_meta_info_per_patient.csv'
    # visual_oob_event_time(anno_meta_info_csv_path)

    
