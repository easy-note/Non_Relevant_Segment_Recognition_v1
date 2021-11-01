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
        self.use_emb = False
        arch_name = self.args.model
        
        model = timm.create_model(arch_name, pretrained=True)
            
        if self.args.model == 'swin_large_patch4_window7_224':
            self.feature_module = nn.Sequential(
                *list(model.children())[:-2],
            )
            self.gap = nn.AdaptiveAvgPool1d(1)
        else:
            self.feature_module = nn.Sequential(
                *list(model.children())[:-1]
            )
        
        if self.args.experiment_type == 'theator':
            for p in self.feature_module.parameters():
                p.requires_grad = False
                
        self.classifier = nn.Linear(model.num_features, 2)
        
        if 'hem-emb' in self.args.hem_extract_mode:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(model.num_features, 2))
        
        
    def forward(self, x):
        features = self.feature_module(x)
        if self.args.model == 'swin_large_patch4_window7_224':
            features = self.gap(features.permute(0, 2, 1))
            
        output = self.classifier(features.view(x.size(0), -1))
        
        if self.use_emb and self.training:
            return features, output
        else:
            return output