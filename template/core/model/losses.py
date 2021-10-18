import torch.nn as nn


name_to_loss = {
    'ce': nn.CrossEntropyLoss,
}


def get_loss(args):
    return name_to_loss[args.loss_fn]()