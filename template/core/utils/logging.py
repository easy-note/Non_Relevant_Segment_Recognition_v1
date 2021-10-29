import time
import os
import pandas as pd
import json
import numpy as np

from core.utils.parser import FileLoader # file load helper



# use in Report class (def save_report) : save json file for numpy floats (should casting)
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


class LogHelper():
    """
    logging -> 추후 log library 적용(log 종류에 따라 형식이 약간 다름?)
    log 는 바로바로 write 할 때 마다 작성
    """

    def __init__(self, save_dir):
        f_dir, target_file = os.path.split(save_dir)
        f_name, ext = os.path.splitext(target_file)

        self.save_dir = save_dir
        self.save_path = os.path.join(f_dir, '{}_{}{}'.format(f_name, self.get_current_time()[0], ext))

        print('=========> SAVING LOG ... | {}'.format(self.save_path)) # init print
    
    def writeln(self, log_txt=""):
        if log_txt != "" :
            logging = '{}\t\t{}'.format(self.get_current_time()[1], log_txt)
        else :
            logging = log_txt
        
        logging += '\n'
        
        self.save(logging)

    def get_current_time(self):
        startTime = time.time()
        s_tm = time.localtime(startTime)
        
        return time.strftime('%Y-%m-%d-%H:%M:%S', s_tm), time.strftime('%Y-%m-%d %I:%M:%S %p', s_tm)

    # save log txt
    def save(self, logging):
        with open(self.save_path, 'a') as f :
            f.write(logging)


class Report():
    def __init__(self, report_save_path):
        self.report_save_path = report_save_path # .json

        self.total_report = self._init_total_report_form()

        self.experiment = self.total_report['experiment']
        self.patients_report = self.total_report['experiment']['patients'] # []
        self.videos_report = self.total_report['experiment']['videos'] # []
        

    def _init_total_report_form(self):
        init_total_report_form = {
            'experiment': {
                'model':'',
                'method':'',
                'inference_fold':'',
                'mCR':0,
                'mOR':0,
                'CR':0,
                'OR':0,
                'details_path':'',

                'patients':[],
                'videos':[],
            },
        }

        return init_total_report_form
    
    def _get_report_form(self, report_type):
        init_report_form = { # one-columne of experiments report
            'patient': { # one-columne of inference report (each patients)
                'patient_no' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
                'gt_IB':0,
                'gt_OOB':0,
                'predict_IB':0,
                'predict_OOB':0,
            },

            'video': { # one-columne of inference report (each videos)
                'patient_no' : '',
                'video_no' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
                'gt_IB':0,
                'gt_OOB':0,
                'predict_IB':0,
                'predict_OOB':0,
            }
        }

        return init_report_form[report_type]
    
    def set_report_save_path(self, report_save_path):
        self.report_save_path = report_save_path
    
    def set_experiment(self, model, methods, inference_fold, mCR, mOR, CR, OR, details_path):
        self.experiment['model'] = model
        self.experiment['methods'] = methods
        self.experiment['infernce_fold'] = inference_fold
        self.experiment['mCR'] = mCR
        self.experiment['mOR'] = mOR
        self.experiment['CR'] = CR
        self.experiment['OR'] = OR
        # TODO - patients.. 
        self.experiment['details_path'] = details_path
    
    def add_patients_report(self, patient_no, FP, TP, FN, TN, TOTAL, CR, OR, gt_IB, gt_OOB, predict_IB, predict_OOB):
        patient = self._get_report_form('patient')

        patient['patient_no'] = patient_no
        patient['FP'] = FP
        patient['TP'] = TP
        patient['FN'] = FN
        patient['TN'] = TN
        patient['TOTAL'] = TOTAL
        patient['CR'] = CR
        patient['OR'] = OR
        patient['gt_IB'] = gt_IB
        patient['gt_OOB'] = gt_OOB
        patient['predict_IB'] = predict_IB
        patient['predict_OOB'] = predict_OOB
        
        self.patients_report.append(patient)

        return patient

    def add_videos_report(self, patient_no, video_no, FP, TP, FN, TN, TOTAL, CR, OR, gt_IB, gt_OOB, predict_IB, predict_OOB):
        video = self._get_report_form('video')

        video['patient_no'] = patient_no
        video['video_no'] = video_no
        video['FP'] = FP
        video['TP'] = TP
        video['FN'] = FN
        video['TN'] = TN
        video['TOTAL'] = TOTAL
        video['CR'] = CR
        video['OR'] = OR
        video['gt_IB'] = gt_IB
        video['gt_OOB'] = gt_OOB
        video['predict_IB'] = predict_IB
        video['predict_OOB'] = predict_OOB
        
        self.videos_report.append(video)
        
        return video

    def clean_report(self):
        self.total_report = self._init_total_report_form()

        self.experiment = self.total_report['experiment']
        self.patients_report = self.total_report['experiment']['patients'] # []
        self.videos_report = self.total_report['experiment']['videos'] # []
    
    def load_report(self):
        if os.path.isfile(self.report_save_path):
            f_loader = FileLoader()
            f_loader.set_file_path(self.report_save_path)
            saved_report_dict = f_loader.load()        

        self.experiment = saved_report_dict['experiment']
        self.patients_report = saved_report_dict['experiment']['patients']
        self.videos_report = saved_report_dict['experiment']['videos']

        self.total_report = self._init_total_report_form()

    def save_report(self):
        json_string = json.dumps(self.total_report, indent=4, cls=MyEncoder)
        print(json_string)

        with open(self.report_save_path, "w") as json_file:
            json.dump(self.total_report, json_file, indent=4, cls=MyEncoder)

    def get_patients_CR(self):
        patients_CR = {}

        for patient in self.total_report['experiment']['patients']:
            patient_no = patient['patient_no']
            patient_CR = patient['CR']
            patients_CR[patient_no] = patient_CR

        return patients_CR

    def get_patients_OR(self):
        patients_OR = {}

        for patient in self.total_report['experiment']['patients']:
            patient_no = patient['patient_no']
            patient_OR = patient['OR']
            patients_OR[patient_no] = patient_OR

        return patients_OR
        

# NOT USED
class ReportHelper():
    # inference_main return으로 experiments 결과 저장, 저장하기 위해 experiments results sheet(csv) path가 args에 포함되어 있어야 하지 않을까?
    def __init__(self, report_save_path):
        self.report_save_path = report_save_path 
        
        self.total_report = self._init_total_report_form()
        

    def _report_type_sanity_check(self, report_type):
        support_report_type = ['experiments', 'patients', 'videos']
        assert report_type in support_report_type, 'NOT SUPPORT REPORT TYPE'

    def _report_form_sanity_check(self, report_type, report_form):
        assert set(self.get_report_form(report_type).keys()) == set(report_form.keys()), 'report form is not sanity'

    def _init_total_report_form(self):
        init_total_report_form = {
            'experiments': {
                'patients':[],
                'videos':[],
            },
        }

    def add_report(self, report_type, report):
        self._report_type_sanity_check(report_type)
        self._report_form_sanity_check(report_type, report) # sanity check : input(report_form)'s keys should be same as init_report_forms's keys

        if report_type == 'experiments':
            self.total_report[report_type] = report
        
        else:
            self.total_report['experiments'][report_type].append(report)


    def get_report_form(self, report_type):
        self._report_type_sanity_check(report_type)

        init_report_form = { # one-columne of experiments report
            'experiments': {
                'model':'',
                'method':'',
                'inference_fold':'',
                'mCR':0,
                'mOR':0,
                'CR':0,
                'OR':0,
            },

            'patients': { # one-columne of inference report (each patients)
                'patient_no' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
                'gt_IB':0,
                'gt_OOB':0,
                'predict_IB':0,
                'predict_OOB':0,
            },

            'videos': { # one-columne of inference report (each videos)
                'patient_no' : '',
                'video_no' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
                'gt_IB':0,
                'gt_OOB':0,
                'predict_IB':0,
                'predict_OOB':0,
            }
        }

        return init_report_form[report_type]

    def save_report(self, report_type, report_form):
        self._report_form_sanity_check(report_type, report_form) # sanity check : input(report_form)'s keys should be same as init_report_forms's keys
        report_df = pd.DataFrame.from_dict([report_form]) # dict to df
        report_df = report_df.reset_index(drop=True)

        merged_df = report_df
        if os.path.isfile(self.report_save_path): # append
            f_loader = FileLoader()
            f_loader.set_file_path(self.report_save_path)
            saved_report_df = f_loader.load()

            saved_report_df.drop(['Unnamed: 0'], axis = 1, inplace = True) # to remove Unmaned : 0 colume

            merged_df = pd.concat([saved_report_df, report_df], ignore_index=True, sort=False)
            
        merged_df.to_csv(self.report_save_path, mode='w')

        print(merged_df)

        return report_df