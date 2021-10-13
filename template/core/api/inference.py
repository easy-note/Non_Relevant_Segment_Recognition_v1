from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from core.dataset.test_dataset import DBDataset, IntervalSampler

class InferenceDB(): # InferenceDB
    """
        Inference per DB from model
    """ 

    def __init__(self, model, db_path, inference_interval=1):
        self.model = model
        self.model = self.model.eval()

        self.db_path = db_path

        self.inference_interval = inference_interval

    def set_db_path(self, db_path):
        self.db_path = db_path

    def set_inference_interval(self, inference_interval):
        self.inference_interval = inference_interval

    def _gen_dataset(self):
        return DBDataset(self.db_path)

    def _gen_sampler(self):
        return IntervalSampler(self._gen_dataset(), interval=self.inference_interval)

    def start(self): 
        print('\n\t########## INFERENCEING (DB) #########\n')

        # DB dataset
        print('DB PATH : \t\t\t{}'.format(self.db_path))
        print('INFERENCE_INTERVAL : \t\t{}'.format(self.inference_interval))

        # data loder
        dl = DataLoader(dataset=self._gen_dataset(),
                        sampler=self._gen_sampler(),
                        batch_size=128)
        
        # inference log
        predict_list = []
        target_img_list = []
        target_frame_idx_list = []

        # inferencing model
        with torch.no_grad() :
            for sample in tqdm(dl, desc='Inferencing... \t ==> {}'.format(self.db_path)) :
                batch_input = sample['img'].cuda()
                batch_output = self.model(batch_input)

                # predict
                batch_predict = torch.argmax(batch_output.cpu(), 1)
                batch_predict = batch_predict.tolist()

                # save results
                predict_list += list(batch_predict)
                target_img_list += sample['img_path'] # target img path
                target_frame_idx_list += sample['db_idx'] # target DB

        target_frame_idx_list = list(map(int, target_frame_idx_list)) # '0000000001' -> 1

        return predict_list, target_img_list, target_frame_idx_list