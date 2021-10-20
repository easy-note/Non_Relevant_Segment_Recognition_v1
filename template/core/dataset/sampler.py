import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def oversampler(labels):
    labels = torch.Tensor(labels)
    uni_label = torch.unique(labels) 
    uni_label_sorted, _  = torch.sort(uni_label)
    uni_label_sorted = uni_label_sorted.detach().numpy()
    label_bin = torch.bincount(labels.int()).detach().numpy()
    sample_weights = 1. / label_bin

    w_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return w_sampler


class OverSampler():
    # 이상하게 1번만 돌고 자꾸 멈춤...
    def __init__(self, labels, batch_size):
        self.labels = labels
        labels = torch.Tensor(labels)
        self.batch_size = batch_size
        
        self.indices = list(range(len(labels)))
        uni_label = torch.unique(labels) 
        
        uni_label_sorted, _  = torch.sort(uni_label)
        uni_label_sorted = uni_label_sorted.detach().numpy()
        label_bin = torch.bincount(labels.int()).detach().numpy()
        label_to_count = dict(zip(uni_label_sorted , label_bin))
        weights = [ len(labels) / label_to_count[float(label)] for label in labels]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.batch_size, replacement=True))

    def __len__(self):
        return len(self.labels)