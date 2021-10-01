import pandas as pd
import numpy as np
import json

class evalHelper:
    def __init__(self, model_output_csv_path:str, gt_json_path:str, inference_step:int):
        self.model_output_csv_path = model_output_csv_path
        self.gt_json_path = gt_json_path
        self.inference_step = inference_step

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

        gt_idx_list=[]
        gt_return_list=[]

        # open annotation json file
        with open(self.gt_json_path) as json_file :
            json_data = json.load(json_file)
            totalframe = json_data['totalFrame'] # totalframe = 51540

            framerate = json_data['frameRate']

            # annotation frame
            for anno_data in json_data['annotations'] :
                start = anno_data['start'] # start frame number
                end = anno_data['end'] # end frame number

                gt_idx_list.append([start, end]) # gt_idx_list = [[start, end], [start, end]..]

        gt_list=[0]*totalframe # gt_list = [0, 0, 0, 0, 0, 0, ... , 0]

        for gt_idx in gt_idx_list:
            ## if last annotation frame number exceeds the total number of frames, 
            ## cut the last annotation frame to the total number of frames. 
            if gt_idx[1] >= totalframe :
                gt_idx[1] = totalframe-1

            # indicate OOB (1)
            for i in range(gt_idx[0], gt_idx[1]+1) :
                gt_list[i]=1

        # convert gt_list to inference step.
        for i, gt in enumerate(gt_list) :
            if i % self.inference_step==0 :
                gt_return_list.append(gt)

        ## FFmpeg으로 1초에 1개씩 프레임을 추출하면 소수점 이하는 버려져야 (truncate, *NOT* round) 함. --> 아닌 경우도 있음 (R_100_ch1)
        # if totalframe % framerate != 0:
        #     gt_return_list.pop(-1)

        return gt_return_list # gt_return_list = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, ... , 0]

    def load_gt_list_predict_list(self):
    
        ## 1. parity check - file format 
        if not self.file_format_parity_check():
            raise Exception('ERROR: file format is not csv or json')

        ## mobile
        # read model_output_list as pandas.
        model_output_result=pd.read_csv(self.model_output_csv_path)
        # convert pandas to list format.
        model_output_list=list(np.array(model_output_result['predict'].tolist()))

        # read gt_list from annotation json file.
        gt_list=self.convert_gt_json()

        # gt_list & model_output_list sanity check
        ### 만약 gt_list 가 model_output_list 보다 더 생성되었다면, model_output_list 길이만큼 앞에서부터 자르기. 
        ### 만약 gt_list 보다 model_output_list 가 더 많이 생성되었다면 -> 문제 있음 (에러 발생).
        if len(gt_list) > len(model_output_list):
            gt_list = gt_list[:len(model_output_list)]
        elif len(gt_list) < len(model_output_list):
            raise Exception('ERROR: gt_list < model_output_list \n====>\t {}, {}'.format(len(gt_list), len(model_output_list)))
        
        return gt_list, model_output_list