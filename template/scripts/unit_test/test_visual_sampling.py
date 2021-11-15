def visual_flow_for_sampling(final_assets_df, save_dir):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    from core.utils.prepare import PatientsGT
    from core.api.visualization import VisualTool
    
    print('final_assets_df')
    print(final_assets_df)

    def get_patient_no(img_db_path):
        cleand_file_name = os.path.splitext(os.path.basename(img_db_path))[0]
        file_info, frame_idx = cleand_file_name.split('-')
        
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no = file_info.split('_')
        patient_no = '_'.join([op_method, patient_idx])

        return patient_no

    def get_frame_idx(img_db_path):
        cleand_file_name = os.path.splitext(os.path.basename(img_db_path))[0]
        file_info, frame_idx = cleand_file_name.split('-')

        return int(frame_idx)
    
    def split_hem_vanila(assets_df):
        HEM_CLASS, NON_HEM_CLASS = (0,1)
        RS_CLASS, NRS_CLASS = (0,1)
        split_assets = {
            'neg_hard_idx':[],
            'pos_hard_idx':[],
            'neg_vanila_idx':[],
            'pos_vanila_idx':[],
        }

        split_assets['neg_hard_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == HEM_CLASS) & (assets_df['class_idx'] == RS_CLASS)]['frame_idx']).tolist()
        split_assets['pos_hard_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == HEM_CLASS) & (assets_df['class_idx'] == NRS_CLASS)]['frame_idx']).tolist()
        split_assets['neg_vanila_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == NON_HEM_CLASS) & (assets_df['class_idx'] == RS_CLASS)]['frame_idx']).tolist()
        split_assets['pos_vanila_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == NON_HEM_CLASS) & (assets_df['class_idx'] == NRS_CLASS)]['frame_idx']).tolist()

        return split_assets

    
    final_assets_df['patinet_no'] = final_assets_df['img_path'].apply(get_patient_no) # extract patinet_no from image db path
    final_assets_df['frame_idx'] = final_assets_df['img_path'].apply(get_frame_idx) # extract frame index from image db path

    patients_assets = PatientsGT()    
    grouped = final_assets_df.groupby('patinet_no') # grouping by patinets no

    # visualziation
    for key, group in tqdm(grouped, desc='Sampling Visualization'): # per each patient
        print('Patient:{} - Sampling: {}'.format(key, len(group)))
        
        # get patinets gt
        patient_gt = patients_assets.get_gt(patient_no=key)

        # parsing hem/vanila assets info
        split_assets = split_hem_vanila(group)

        # visual
        visual_tool = VisualTool(gt_list=patient_gt, patient_name=key, save_path=os.path.join(save_dir, 'sampling-{}.png'.format(key)))
        visual_tool.visual_sampling(split_assets['neg_hard_idx'], split_assets['pos_hard_idx'], split_assets['neg_vanila_idx'], split_assets['pos_vanila_idx'], model_name='mobiletnet', window_size=9000, section_num=2)


def main():
    import pandas as pd

    final_assets_path = '/OOB_RECOG/logs/211023_TRAIN_HEM-softmax-FOLD1-trial:1-fold:1/TB_log/version_9/hem_assets.csv'
    final_assets_df = pd.read_csv(final_assets_path)
    save_dir = '/OOB_RECOG/results_sampling'

    # visual sampling
    # annotation_parser = AnnotationParser('/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V3/TBE/01_G_01_R_100_ch1_01_TBE_30.json')
    # gt_list = annotation_parser.get_event_sequence(extract_interval=1)

    print(final_assets_df)
    visual_flow_for_sampling(final_assets_df, save_dir)

if __name__ == '__main__':
    
    if __package__ is None:
        import sys
        import os
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(base_path)

    main()

