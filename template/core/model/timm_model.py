import timm
import torch
import torch.nn as nn


def generate_timm_model(args):
    model = TIMM(args)
    
    return model


class TIMM(nn.Module):
    """
        SOTA model usage
    """
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        arch_name = self.args.model
        
        self.model = timm.create_model(arch_name, pretrained=True)
        
        if arch_name == 'ig_resnext101_32x48d':
            self.model.fc = nn.Linear(2048, 2)
        elif arch_name == 'swin_large_patch4_window7_224':
            self.model.head = nn.Linear(1536, 2)
            
        
        
        
    def forward(self, x):
        return self.model(x)