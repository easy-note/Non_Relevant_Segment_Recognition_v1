import os
import csv
import pandas as pd
import numpy as np

import json
import argparse

from core.utils.evalHelper import evalHelper # eval Helper
import OOB_Post_Procssing_module # PP Module

parser = argparse.ArgumentParser()

parser.add_argument('--model_output_csv_path', type=str, 
                    default='/OOB_RECOG/shared_OOB_Inference_Module/assets/Inference/Inference-ROBOT-01_G_01_R_100_ch1_01.csv', help='model predict output file path')
parser.add_argument('--gt_json_path', type=str, 
                    default='/OOB_RECOG/shared_OOB_Inference_Module/assets/Annotation(V2)/01_G_01_R_100_ch1_01_OOB_27.json', help='ground-truth (annotation) file path')

parser.add_argument('--inference_step', type=int, default=30, help='inference interval')

args, _ = parser.parse_known_args()

class Inference_eval:
    def __init__(self, model_output_csv_path:str, gt_json_path:str, inference_step:int):
        self.model_output_csv_path = model_output_csv_path
        self.gt_json_path = gt_json_path
        self.inference_step = inference_step

        self.video_name = '_'.join(self.gt_json_path.split('/')[-1].split('.')[0].split('_')[:7])
        
    def create_directory(self, dir):
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("ERROR: Failed to created the directory")

    def list_length_parity_check(self, model_output_list:list, gt_list:list):
        """
        Check the the number of model_output_list match the number of gt_list.

        Args:
            model_output_list: `list`, model predict output list.
            gt_list: `list`, gt_list.

        Returns:
            True or False (Boolean)
        """

        return len(model_output_list) != len(gt_list)

    def file_format_parity_check(self):
        return ('csv' in self.model_output_csv_path) and ('json' in self.gt_json_path)

    def convert_gt_json(self):
        """
        Read groud-truth(gt) json file, covert gt json to binary list.

        Returns:
            `list` of groud-truth. 

        Example:
            >>> convert_gt_json()
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, ... , 0]
        """

        self.gt_idx_list=[]
        self.gt_return_list=[]

        # open annotation json file
        with open(self.gt_json_path) as self.json_file :
            self.json_data = json.load(self.json_file)
            self.totalframe = self.json_data['totalFrame'] # totalframe = 51540

            # annotation frame
            for self.anno_data in self.json_data['annotations'] :
                self.start = self.anno_data['start'] # start frame number
                self.end = self.anno_data['end'] # end frame number

                self.gt_idx_list.append([self.start, self.end]) # gt_idx_list = [[start, end], [start, end]..]

        self.gt_list=[0]*self.totalframe # gt_list = [0, 0, 0, 0, 0, 0, ... , 0]

        for self.gt_idx in self.gt_idx_list:
            ## if last annotation frame number exceeds the total number of frames, 
            ## cut the last annotation frame to the total number of frames. 
            if self.gt_idx[1] >= self.totalframe :
                self.gt_idx[1] = self.totalframe-1

            # indicate OOB (1)
            for self.i in range(self.gt_idx[0], self.gt_idx[1]+1) :
                self.gt_list[self.i]=1

        # convert gt_list to inference step.
        for i, gt in enumerate(self.gt_list) :
            if i % self.inference_step==0 :
                self.gt_return_list.append(gt)

        return self.gt_return_list # gt_return_list = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, ... , 0]

    def calc_OR_CR(self):
        """
        Calculate Over_Estimation_Ratio(OR) and Confidence_Ratio(CR).
        
        Save results (OR, CR) to json object.
        output json format(str) : results-OR_CR
        {
            "01_G_01_R_100_ch1_03": {
                "over_estimation_ratio": 0.01046610685164902,
                "confidence_ratio": 0.9785809906291834
                "FP": 20,
                "FN": 1,
                "TP": 150,
                "TN": 4630,
                "TOTAL": 4881
            }
        }

        Returns:
            str(json format)

        Example:
            >>> calc_OR_CR()
            str(json format)
        """

        ## 1. parity check - file format 
        if not self.file_format_parity_check():
            raise Exception('ERROR: file format is not csv or json')

        # read model_output_list as pandas.
        self.model_output_result=pd.read_csv(self.model_output_csv_path)
        # convert pandas to list format.
        self.model_output_list=list(np.array(self.model_output_result['predict'].tolist()))

        ####### Post Processing #######
        ##### example 1. when you want to apply best pp filter (default sequence fps = 1)
        fb = OOB_Post_Procssing_module.FilterBank(self.model_output_list, seq_fps=1)
        self.model_output_list = fb.apply_best_filter() # modify after pp
        ####### Post Processing #######

        # read gt_list from annotation json file.
        self.gt_list=self.convert_gt_json()

        ### 21.09.28 HG 추가. for matchihng length of predict list and gt list 
        print('ORIGIN : len(gt_list) {}'.format(len(self.gt_list)))

        ##### evaluation Helper [strat] #####
        eval_helper = evalHelper(self.model_output_csv_path, self.gt_json_path, self.inference_step)
        self.gt_list, self.model_output_list = eval_helper.load_gt_list_predict_list()
        ##### evaluation Helper [end] #####

        print('MODIFIED : len(gt_list) {}'.format(len(self.gt_list)))
        
        ## 2. parity check - list length 
        ## if len(model_output_list) != len(gt_list) -> raise ERROR, exit(1)
        if self.list_length_parity_check(self.model_output_list, self.gt_list):
            raise Exception('ERROR: the number of model_output_list does not match the number of gt_list.')

                # calculate TP, TN, FP, FN
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        for self.predict, self.gt in zip(self.model_output_list, self.gt_list) :
            if self.predict == self.gt :
                if self.predict == 0 :
                    self.TN += 1
                elif self.predict == 1 :
                    self.TP += 1
            elif self.predict != self.gt :
                if self.predict == 0 :
                    self.FN += 1
                elif self.predict == 1 :
                    self.FP += 1

        # caculate over_estimation_ratio(OR), confidence_ratio(CR)
        try:
            self.over_estimation_ratio=self.FP/(self.FP+self.TP+self.FN)
            self.confidence_ratio=(self.TP-self.FP)/(self.FP+self.TP+self.FN)
        
        ## if (FP+TP+FN)==0, set over_estimation_ratio, confidence_ratio as 1
        except: 
            self.over_estimation_ratio=-1
            self.confidence_ratio=-1

        # HG.21.10.14 HG. json file write 대신 json object return
        calc_OR_CR_dict = {}
        calc_OR_CR_dict[self.video_name] = {'over_estimation_ratio':self.over_estimation_ratio,
                                            'confidence_ratio':self.confidence_ratio,
                                            'FP':self.FP,
                                            'FN':self.FN,
                                            'TP':self.TP,
                                            'TN':self.TN,
                                            'TOTAL':self.TP + self.TN + self.FP + self.FN} 

        metric_json = json.dumps(calc_OR_CR_dict, indent=2)
        print(metric_json)

        return metric_json # str(json format)

if __name__ == '__main__':
    test = Inference_eval(model_output_csv_path=args.model_output_csv_path, gt_json_path=args.gt_json_path, inference_step=args.inference_step)
    test.calc_OR_CR()