import time
import os
import pandas as pd

from core.utils.parser import FileLoader # file load helper

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


class ReportHelper():
    # inference_main return으로 experiments 결과 저장, 저장하기 위해 experiemnts results sheet(csv) path가 args에 포함되어 있어야 하지 않을까?
    def __init__(self, report_save_path, report_type):
        self.report_save_path = report_save_path 
        self.report_type = report_type # .csv

        support_report_type = ['experiments', 'patients', 'videos']
        assert report_type in support_report_type, 'NOT SUPPORT REPORT TYPE'

    def _report_form_sanity_check(self, report_form):
        assert set(self.get_report_form().keys()) == set(report_form.keys()), 'report form is not sanity'

    def get_report_form(self):
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
                'Patient' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
            },

            'videos': { # one-columne of inference report (each videos)
                'Patient' : '',
                'Video' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
            }
        }

        return init_report_form[self.report_type]

    def save_report(self, report_form):
        self._report_form_sanity_check(report_form) # sanity check : input(report_form)'s keys should be same as init_report_forms's keys
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
