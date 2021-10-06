from torch.utils.data import DataLoader, BatchSampler
import torch
from tqdm import tqdm

from core.dataset.test_dataset import DBDataset, IntervalSampler

class InferenceDB(): # InferenceDB
    """
        Inference per DB from model
    """ 

    def __init__(self, model, db_path, inference_interval=30):
        self.model = model
        self.model = self.model.eval()

        self.db_path = db_path

        self.test_dataset = DBDataset(db_path)

        self.batch_sampler = self.get_batch_sampler(inference_interval) # default interval = 30

    def set_inference_interval(self, inference_interval):
        self.batch_sampler = self.get_batch_sampler(inference_interval)

    def get_batch_sampler(self, inference_interval):
        return BatchSampler(sampler=IntervalSampler(self.test_dataset, interval=inference_interval),
                            batch_size=128, drop_last=False) # default batch_size=128

    def start(self): 
        print('\n\t########## INFERENCEING (DB) #########\n')

        # DB dataset
        print('DB PATH : ', self.db_path)

        # data loder
        dl = DataLoader(dataset=self.test_dataset,
                        batch_sampler=self.batch_sampler)
        
        # inference log
        predict_list = []
        target_img_list = []
        target_frame_idx_list = []

        cnt = 0
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
                target_frame_idx_list += sample['db_idx'] # target DB index

        target_frame_idx_list = list(map(int, target_frame_idx_list)) # '0000000001' -> 1

        return predict_list, target_img_list, target_frame_idx_list