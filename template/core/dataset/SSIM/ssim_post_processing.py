
import os
import glob
import natsort
import pandas as pd
from itertools import groupby
import numpy as np

def encode_list(s_list): # run-length encoding from list
    return [[len(list(group)), key] for idx, (key, group) in enumerate(groupby(s_list))] # [[length, value], [length, value]...]

def cumsum(assets_df):
    class_list = assets_df['class'].values.tolist()
    encode_data = pd.DataFrame(data=encode_list(class_list), columns=['length', 'class']) # [length, value]
        
    # print('\nencode_data')
    # print(encode_data)

    # arrange data
    runlength_df = pd.DataFrame(range(0,0)) # empty df
    runlength_df = runlength_df.append(encode_data)

    # print('\nrunlength_df')
    # print(runlength_df)

    # Nan -> 0, convert to int
    runlength_df = runlength_df.fillna(0).astype(int)

    # split data, class // both should be same length
    runlength_class = runlength_df['class'] # class info
    runlength_model = runlength_df['length'] # run length info of model prediction

    # data processing for barchart
    data = np.array(runlength_model.to_numpy()) # width
    data_cum = data.cumsum(axis=0) # for calc start index, 누적 합

    starts_list = []
    for i, frame_class in enumerate(runlength_class) :
        widths = data[i]
        starts_list.append(data_cum[i] - widths)

    runlength_df['st_pos'] = starts_list
    runlength_df['ed_pos'] = data_cum

    # print('\runlength_df')
    # print(runlength_df)

    return runlength_df



def read_files(target_name, target_fps):
    pd.set_option('display.max_rows', None)

    base_path = '/raid/SSIM_RESULT/0.997-SSIM_RESULT'

    target_path = os.path.join(base_path, target_name)

    target_files = glob.glob(os.path.join(target_path, '*', '*-{}FPS.csv').format(target_fps))
    target_files = natsort.natsorted(target_files)

    for target_file in target_files:

        if target_file.split('/')[-1].split('-')[0] == 'pp':
            continue

        print('\ntarget_file : ', target_file)
        asset_df = pd.read_csv(target_file)
        cumsum_asset_df = cumsum(asset_df)

        pp_asset_df = cumsum_asset_df[(cumsum_asset_df['class']==3) & (cumsum_asset_df['length']>target_fps) | ((cumsum_asset_df['class']==1) & (cumsum_asset_df['length']>target_fps))]
        
        print(pp_asset_df)
        
        pp_list = []
        for idx, row in pp_asset_df.iterrows():
            st_pos = row['st_pos']
            length = row['length']

            for i in range(st_pos+1, st_pos+length+1):
                pp_list.append(i)

        asset_df = asset_df.drop(pp_list)

        f_name = 'pp-' + '.'.join(target_file.split('/')[-1].split('.')[:-1])
        asset_df.to_csv(os.path.join('/'.join(target_file.split('/')[:-1]), '{}.csv'.format(f_name)))


def asset_concat(target_name, target_fps):
    base_path = '/raid/SSIM_RESULT/0.997-SSIM_RESULT'

    target_path = os.path.join(base_path, target_name)

    target_files = glob.glob(os.path.join(target_path, '*', 'pp-*-{}FPS.csv').format(target_fps))
    target_files = natsort.natsorted(target_files)
    
    df_list = []
    for target_file in target_files:
        data = pd.read_csv(target_file)
        df_list.append(data)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all[['frame_path', 'gt']]

    print(df_all)

    df_all.to_csv(os.path.join(target_path, 'PP-assets-{}-{}FPS.csv'.format(target_name, target_fps)), header=False)

def concat_all(target_fps):
    base_path = '/raid/SSIM_RESULT/0.997-SSIM_RESULT'
    target_files = glob.glob(os.path.join(base_path, '*', 'PP-*{}FPS.csv').format(target_fps))
    target_files = natsort.natsorted(target_files)

    df_list = []
    for i in target_files:
        data = pd.read_csv(i, header=None)
        df_list.append(data)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.drop(df_all.columns[[0]], axis=1)  # df.columns is zero-based pd.Index 
    print(df_all)

    df_all.to_csv(os.path.join(base_path, 'PP-assets-VIHUB_ALL-{}FPS.csv'.format(target_fps)), header=False)
    
    # print(df_all)


def read_pp_files(target_name, target_fps):
    base_path = '/raid/SSIM_RESULT/0.997-SSIM_RESULT'

    target_path = os.path.join(base_path, target_name)

    target_files = glob.glob(os.path.join(target_path, '*', 'pp-*-{}FPS.csv').format(target_fps))
    target_files = natsort.natsorted(target_files)

    patient_list = []
    total_list = []
    rs_list = []
    nrs_list = []
    for target_file in target_files:
        data = pd.read_csv(target_file)

        gt_list = data['gt'].tolist()

        total = len(gt_list)
        rs = gt_list.count(0)
        nrs = gt_list.count(1)

        print(total, rs, nrs)
        patient_list.append(target_file.split('/')[-1].split('-')[1])
        total_list.append(total)
        rs_list.append(rs)
        nrs_list.append(nrs)
    
    data = {
        'patient_list': patient_list,
        'total_cnt': total_list,
        'rs_cnt': rs_list,
        'nrs_cnt': nrs_list
    }

    df = pd.DataFrame(data)
    df.to_csv('./{}-{}FPS.csv'.format(target_name, target_fps))

        
if __name__ == '__main__':
    # ## read files
    # read_files(target_name='gangbuksamsung_127case', target_fps=5)
    # read_files(target_name='gangbuksamsung_127case', target_fps=1)

    # read_files(target_name='severance_1st', target_fps=5)
    # read_files(target_name='severance_1st', target_fps=1)

    # read_files(target_name='severance_2nd', target_fps=5)
    # read_files(target_name='severance_2nd', target_fps=1)

    # ## asset concat
    # asset_concat(target_name='gangbuksamsung_127case', target_fps=5)
    # asset_concat(target_name='gangbuksamsung_127case', target_fps=1)

    # asset_concat(target_name='severance_1st', target_fps=5)
    # asset_concat(target_name='severance_1st', target_fps=1)
    
    # asset_concat(target_name='severance_2nd', target_fps=5)
    # asset_concat(target_name='severance_2nd', target_fps=1)

    # ## concat all
    # concat_all(target_fps=5)
    # concat_all(target_fps=1)

    # ## read pp files
    read_pp_files(target_name='gangbuksamsung_127case', target_fps=5)
    read_pp_files(target_name='gangbuksamsung_127case', target_fps=1)
    
    read_pp_files(target_name='severance_1st', target_fps=5)
    read_pp_files(target_name='severance_1st', target_fps=1)
    
    read_pp_files(target_name='severance_2nd', target_fps=5)
    read_pp_files(target_name='severance_2nd', target_fps=1)
