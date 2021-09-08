import torch
import torch.nn as nn



class HEMHelper():
    """
        Help computation ids for Hard Example Mining.
        
    """
    def __init__(self):
        super().__init__()
        
    def set_method(self, method):
        self.method = method

    def compute_hem(self, model, data_loader):
        if self.method == 'hem-softmax':
            return self.hem_softmax_diff(model, data_loader)
        elif self.method == 'hem-vi':
            return self.hem_vi(model, data_loader)
        elif self.method == 'hem-bs':
            return self.hem_batch_sampling(model, data_loader)
        else: # exception
            return None

    def hem_softmax_diff(self, model, data_loader):
        pass

    def hem_vi(self, model, data_loader):
        pass

    def hem_batch_sampling(self, model, data_loader):
        pass