import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class HEMHelper():
    """
        Help computation ids for Hard Example Mining.
        
    """
    def __init__(self):
        super().__init__()
        
    def set_method(self, method):
        self.method = method

    def set_batch_size(self, bsz):
        self.bsz = bsz

    def set_n_batch(self, N):
        self.n_bs = N

    def compute_hem(self, model, dataset):
        if self.method == 'hem-softmax':
            return self.hem_softmax_diff(model, dataset)
        elif self.method == 'hem-vi':
            return self.hem_vi(model, dataset)
        elif self.method == 'hem-bs':
            return self.hem_batch_sampling(model, dataset)
        else: # exception
            return None

    def hem_softmax_diff(self, model, dataset):
        pass

    def hem_vi(self, model, dataset):
        pass

    def hem_batch_sampling(self, model, dataset):
        d_loader = DataLoader(dataset, 
                            batch_size=self.bsz, 
                            shuffle=True,
                            drop_last=True)
        n_pick = self.bsz // (self.n_bs * 2)

        y_hat = None
        y_true = None

        # model.eval()

        for _ in range(self.n_bs):
            x, y = next(iter(d_loader))
            x, y = x.cuda(), y.cuda()

            output = nn.functional.softmax(model(x), -1)
            pred_ids = torch.argmax(output, -1)
            pos_chk = pred_ids == y
            neg_chk = pred_ids != y

            # pos_output = x[pos_chk][:n_pick, ]
            # neg_output = x[neg_chk][:n_pick, ]

            pos_output = output[pos_chk][:n_pick, ]
            neg_output = output[neg_chk][:n_pick, ]

            pos_y = y[pos_chk][:n_pick, ]
            neg_y = y[neg_chk][:n_pick, ]
            
            if y_hat is not None:
                y_hat = torch.cat((y_hat, pos_output), 0)
                y_hat = torch.cat((y_hat, neg_output), 0)
            else:
                y_hat = pos_output
                y_hat = torch.cat((y_hat, neg_output), 0)

            if y_true is not None:
                y_true = torch.cat((y_true, pos_y), 0)
                y_true = torch.cat((y_true, neg_y), 0)
            else:
                y_true = pos_y
                y_true = torch.cat((y_true, neg_y), 0)
        
        # model.train()

        return y_hat, y_true