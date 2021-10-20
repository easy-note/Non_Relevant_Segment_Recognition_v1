import torch
import torch.nn as nn
import torchvision.models as models


def generate_resnet(args):
    arch_name = args.model

    if arch_name == 'resnet18':
        model = models.resnet18(pretrained=True)

    for p in model.parameters():
        p.requires_grad = False
    
    model.fc = torch.nn.Linear(512, 2)

    return model