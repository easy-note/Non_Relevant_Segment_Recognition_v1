from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from core.dataset.test_dataset import DBDataset, IDXSampler

class InferenceDB(): # InferenceDB
    """
        Inference per DB from model
    """ 

    def __init__(self, model, DB_path): # DB_path -> db_path
        self.model = model
        self.DB_path = DB_path

        self.model = self.model.eval()

        self.test_dataset = DBDataset(DB_path)

        self.batch_size = 256 # default
        self.step_of_inference = 30 # default

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_step_of_inference(self, step_size):
        self.step_of_inference = step_size

    def start(self):
        print('\n\t########## INFERENCEING (DB) #########\n')

        # DB dataset
        print('DB PATH : ', self.DB_path)

        # target idx
        target_idx_list = list(range(0, len(self.test_dataset), self.step_of_inference))

        # Set index for batch
        s = IDXSampler(target_idx_list, batch_size=self.batch_size)

        # set dataloader with custom batch sampler
        dl = DataLoader(self.test_dataset, batch_sampler=list(s))
        
        # inference log
        target_img_list = []
        target_frame_idx_list = []
        predict_list = []

        # inferencing model
        with torch.no_grad() :
            for sample in tqdm(dl, desc='Inferencing... \t ==> {}'.format(self.DB_path)) :
                BATCH_INPUT = sample['img'].cuda()
                BATCH_OUTPUT = self.model(BATCH_INPUT)

                # predict
                BATCH_PREDICT = torch.argmax(BATCH_OUTPUT.cpu(), 1)
                BATCH_PREDICT = BATCH_PREDICT.tolist()

                # save results
                predict_list += list(BATCH_PREDICT)
                target_img_list += sample['img_path'] # target img path
                target_frame_idx_list += sample['DB_idx'] # target DB index

        return predict_list
    