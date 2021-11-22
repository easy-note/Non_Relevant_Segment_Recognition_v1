import timm
import torch
import torch.nn as nn


def generate_timm_model(args):
    model = TIMM(args)
    
    return model


class TIMM(nn.Module):
    """
        SOTA model usage
        1. resnet18
        2. repvgg_b0
        3. mobilenetv3_large_100
        
    """
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.use_emb = False
        arch_name = self.args.model
        
        model = timm.create_model(arch_name, pretrained=True)
        
        # help documents - https://fastai.github.io/timmdocs/create_model (how to use feature_extractor in timm)
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
        
        if 'hem-emb' in self.args.hem_extract_mode or 'hem-focus' in self.args.hem_extract_mode:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(model.num_features, 2))
        
        else : # off-line and genral
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.args.dropout_prob, inplace=True),
                nn.Linear(model.num_features, 2)
            )
        
        if self.args.use_online_mcd:
            self.dropout = nn.Dropout(self.args.dropout_prob)
        
    def forward(self, x):
        features = self.feature_module(x)
        
        if self.args.model == 'swin_large_patch4_window7_224':
            features = self.gap(features.permute(0, 2, 1))
            
        if self.args.use_online_mcd: #  해당 방식처럼 training에 따라서 forward 컨셉으로 offline도 구성하고 싶었는데.. 우선 online을 제외한 general. offline에서 classifier에 dropout을 추가하는 방향으로 임시 변경하였습니다.
            if self.training: # 학습 중간에 넣나요? 보규님? 어떻게 접근하는지 궁금합니다!
                features = self.dropout(features)
            else:
                mcd_outputs = []
                for _ in range(self.args.n_dropout):
                    mcd_outputs.append(self.dropout(features).unsqueeze(0))
                    
                a = torch.vstack(mcd_outputs)
                features = torch.mean(a, 0)
            
        output = self.classifier(features.view(x.size(0), -1))
        
        if self.use_emb and self.training:
            return features, output
        else:
            return output