import json
import os
import numpy as np


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# log folder path
base_path_list = [
    # '/code/OOB_Recog/logs-mobilenet',
    #               '/code/OOB_Recog/logs-resnet',
    #               '/code/OOB_Recog/logs-repvgg',
    #               '/code/OOB_Recog/logs-multi',
    #               '/code/OOB_Recog/logs-other-fold',
                  '/code/OOB_Recog/logs-theator',
                  ]

for base_path in base_path_list:
    # logs
    for dir_name in os.listdir(base_path):
        dpath = os.path.join(base_path, dir_name, 'TB_log')
        
        for dir_ver in os.listdir(dpath):
            dpath2 = os.path.join(dpath, dir_ver)
            
            if os.path.exists(dpath2 + '/Report.json'):
                json_path = dpath2 + '/Report.json'
                json_path2 = dpath2 + '/Report2.json'
                
                with open(json_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    
                    patient_data = json_data['experiment']['patients']
                    
                    all_TP = 0
                    all_FP = 0
                    all_FN = 0
                    
                    recall_list = []
                    prec_list = []
                    
                    for p_data in patient_data:
                        all_TP += p_data['TP']
                        all_FP += p_data['FP']
                        all_FN += p_data['FN']
                        
                        recall_list.append( p_data['TP'] / (p_data['TP'] + p_data['FN']) )
                        prec_list.append( p_data['TP'] / (p_data['TP'] + p_data['FP']) )
                    
                    m_rec = sum(recall_list) / len(recall_list)
                    m_prec = sum(prec_list) / len(prec_list)
                    rec = all_TP / (all_TP + all_FN)
                    prec = all_TP / (all_TP + all_FP)
                    
                    # print(len(patient_data), m_rec, m_prec, rec, prec)
                    
                    new_json_data = {
                        'experiment': {}
                    }
                    new_json_data['experiment']['mPrec'] = m_prec
                    new_json_data['experiment']['mRec'] = m_rec
                    new_json_data['experiment']['Prec'] = prec
                    new_json_data['experiment']['Rec'] = rec
                    
                    for key, val in json_data['experiment'].items():
                        new_json_data['experiment'][key] = val
        
                with open(json_path2, 'w') as json_file:
                    json.dump(new_json_data, json_file, indent=4, cls=MyEncoder)