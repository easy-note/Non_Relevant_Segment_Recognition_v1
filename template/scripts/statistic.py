EXCEPTION_NUM = -100

def frame_to_sec(frame, fps):
    return frame * (1/fps)

def statistic_nrs_event_cnt(base_path, save_path):
    import json
    from glob import glob
    import natsort
    import pandas as pd

    from core.utils.parser import InfoParser
    from core.utils.parser import AnnotationParser

    # 0. load all annotation file data
    file_list = natsort.natsorted(glob(os.path.join(base_path, '*.json')))

    # 1. parsing patients
    patient_dict = {}
    info_parser= InfoParser(parser_type='ROBOT_ANNOTATION')

    for fpath in file_list:
        
        # parsing annotation info from file name
        ## ver 1 : using InforParser rule
        '''
        info_parser.write_file_name(fpath)
        patient_no = info_parser.get_patient_no()
        '''

        ## ver 2 : using custom rule
        file_name = os.path.splitext(os.path.basename(fpath))[0] # 04_GS4_99_L_37_01_NRS_30.json
        hospital, department, surgeon, op_method, patient_idx, video_slice_no, _, _ = file_name.split('_') # LAPA Gangbuk annotation rule
        patient_no = '_'.join([surgeon, op_method, patient_idx])
        patient_no = '_'.join([op_method, patient_idx])

        if patient_no in patient_dict:
            patient_dict[patient_no].append(fpath)
        else:
            patient_dict[patient_no] = [fpath]

    # train_patients = [patient + '_' for patient in train_patients_name]
    # val_patients = [patient + '_' for patient in val_patients_name]
    aggregation_cnt = 0 # 환자별처리 count (pateints aggregation)
    aggregation_patients = []

    # init variable
    col_name = ['Patient', 'totalFrame', 'fps', 'RS_count', 'NRS_count', 'total_time', 'RS_event_time', 'NRS_event_time', 'NRS_event_cnt', 'start_frame_idx', 'end_frame_idx', 'start_frame_time', 'end_frame_time', 'NRS_event_duration']
    total_statistic_pd = pd.DataFrame(columns=col_name)

    for key, f_list in patient_dict.items():
        f_list=natsort.natsorted(f_list) # sort for video slice

        # init variable
        patient_totalFrame = 0
        patient_fps = 0
        patient_RS_count = 0
        patient_NRS_count = 0
        patient_NRS_event_cnt = 0

        start_frame_idx = []
        end_frame_idx = []

        for f_path in f_list: # patinets annotation files
            '''
            # ver 1 : using custom
            with open(file_list[fi], 'r') as f:
                data = json.load(f)

            frames = data['totalFrame']
            annos = data['annotations']
            l_anno = len(annos)

            # get annotation info
            for idx, anno in enumerate(annos):
                # exception case
                if fi+1 != len(f_list) and idx+1 == l_anno:
                    with open(file_list[fi+1], 'r') as f:
                            t_data = json.load(f)
                    t_annos = t_data['annotations']
                    if anno['end'] >= frames-1 : # over 
                        t_annos[0]['start'] == 0: 
                            continue # cnt x

                event_cnt[state] += 1
            '''

            # ver 2 : using parsing module
            anno_parser = AnnotationParser(f_path)

            event_sequence = anno_parser.get_event_sequence()
            frames = anno_parser.get_totalFrame()
            annos = anno_parser.get_annotations_info()
            fps = anno_parser.get_fps()

            for idx, anno in enumerate(annos):
                start, end = anno[0] + patient_totalFrame, anno[1] + patient_totalFrame # re calc patients level

                if end >= frames + patient_totalFrame : # adjust to totalFrame
                    end = frames + patient_totalFrame - 1
                
                if idx == 0 and end_frame_idx: # init annotation
                    if end_frame_idx[-1] + 1 == start: # re-set end frame (not append)
                        end_frame_idx[-1] = end
                        
                        aggregation_cnt += 1
                        aggregation_patients.append(key)
                        continue

                start_frame_idx.append(start)
                end_frame_idx.append(end)

            # record
            patient_totalFrame += frames
            patient_fps += fps
            
        # calc nrs frame cnt / nrs event cnt
        for start, end in zip(start_frame_idx, end_frame_idx):
            patient_NRS_count += (end - start) + 1
            patient_NRS_event_cnt += 1
        
        patient_RS_count = patient_totalFrame - patient_NRS_count
        patient_fps = patient_fps / len(f_list)

            # save per patinets
        if patient_NRS_event_cnt == 0:
            patient_statistic_dict = {
                'Patient':key,
                'totalFrame': patient_totalFrame,
                'fps': patient_fps,
                'RS_count': patient_RS_count,
                'NRS_count': patient_NRS_count,
                'total_time': EXCEPTION_NUM,
                'RS_event_time': EXCEPTION_NUM,
                'NRS_event_time': EXCEPTION_NUM,
                'NRS_event_cnt': [0],
                'start_frame_idx': [EXCEPTION_NUM],
                'end_frame_idx': [EXCEPTION_NUM],
                'start_frame_time': EXCEPTION_NUM,
                'end_frame_time': EXCEPTION_NUM,
                'NRS_event_duration': EXCEPTION_NUM,
            }

        else:
            patient_statistic_dict = { 
                'Patient':key,
                'totalFrame': patient_totalFrame,
                'fps': patient_fps,
                'RS_count': patient_RS_count,
                'NRS_count': patient_NRS_count,
                'total_time': EXCEPTION_NUM,
                'RS_event_time': EXCEPTION_NUM,
                'NRS_event_time': EXCEPTION_NUM,
                'NRS_event_cnt': [patient_NRS_event_cnt] * patient_NRS_event_cnt,
                'start_frame_idx': start_frame_idx,
                'end_frame_idx': end_frame_idx,
                'start_frame_time': EXCEPTION_NUM,
                'end_frame_time': EXCEPTION_NUM,
                'NRS_event_duration': EXCEPTION_NUM,
            }
        
        patient_statistic_pd = pd.DataFrame.from_dict(patient_statistic_dict)

        total_statistic_pd = total_statistic_pd.append(patient_statistic_pd, ignore_index=True)        


    # convert time
    total_statistic_pd['total_time'] = total_statistic_pd.apply(lambda x:frame_to_sec(x['totalFrame'], x['fps']), axis=1)
    total_statistic_pd['RS_event_time'] = total_statistic_pd.apply(lambda x:frame_to_sec(x['RS_count'], x['fps']), axis=1)
    total_statistic_pd['NRS_event_time'] = total_statistic_pd.apply(lambda x:frame_to_sec(x['NRS_count'], x['fps']), axis=1)
    total_statistic_pd['start_frame_time'] = total_statistic_pd.apply(lambda x:frame_to_sec(x['start_frame_idx'], x['fps']), axis=1)
    total_statistic_pd['end_frame_time'] = total_statistic_pd.apply(lambda x:frame_to_sec(x['end_frame_idx'], x['fps']), axis=1)
    total_statistic_pd['NRS_event_duration'] = total_statistic_pd['end_frame_time'] - total_statistic_pd['start_frame_time'] + 1/total_statistic_pd['fps'] # why add? corresponding to meaning of time.

    total_statistic_pd.to_csv(save_path)

    print(total_statistic_pd)

    print('\n\n\t\t===== {} =====\t\t\n\n'.format('결과'))
    print('처리된환자개수: {}'.format(len(patient_dict)))
    print('처리환자: {}'.format(list(patient_dict.keys())))

    print('환자별중복처리개수(aggregation): {}'.format(aggregation_cnt)) # 환자별처리 count (pateints aggregation)
    print('중복처리환자(aggregation): {}'.format(aggregation_patients)) # 환자별처리 count (pateints aggregation)

if __name__ == '__main__':
    if __package__ is None:
        import sys
        import os
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core/accessory/RepVGG')
        # print(base_path)

        from core.config.assets_info import annotation_path
        
        # base_path = annotation_path['annotation_v3_base_path']
        base_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS'
        # base_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_1st/NRS'
        # base_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_2nd/NRS'
        
        save_dir = os.path.join('/OOB_RECOG', 'statistic')
        os.makedirs(save_dir, exist_ok=True)
        # statistic_nrs_event_cnt(base_path, os.path.join(save_dir, 'robot_v3_nrs_statistic.csv'))
        statistic_nrs_event_cnt(base_path, os.path.join(save_dir, 'gangbuksamsung_v3_nrs_statistic.csv'))
        # statistic_nrs_event_cnt(base_path, os.path.join(save_dir, 'severance_1st_nrs_statistic.csv'))
        # statistic_nrs_event_cnt(base_path, os.path.join(save_dir, 'severance_2nd_nrs_statistic.csv'))