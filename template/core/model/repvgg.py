import torch
import torch.nn as nn
from core.accessory.RepVGG.repvgg import RepVGG


def generate_repvgg(args):
    model = CustomRepVGG(args)
    
    if args.restore_path is not None:
        import os, glob, natsort

        ver = int(args.restore_path.split('/')[-1].split('_')[-1])
        
        if ver > 0:
            ver -= 1

        t_path = os.path.join(*args.restore_path.split('/')[:-1])
        ckpoint_path = glob.glob(t_path + '/TB_log/version_{}/checkpoints/*.pt'.format(ver))
        
        if len(ckpoint_path) > 0:
            print(ckpoint_path)
            ckpts = glob.glob(ckpoint_path)
            ckpts = natsort.natsorted(ckpts)
            
            model.load_state_dict(torch.load(ckpts[-1]))
            
    return model



class CustomRepVGG(nn.Module):
    """
        RepVGG custom
        ref : https://github.com/DingXiaoH/RepVGG
    """
    def __init__(self, args):
        super(CustomRepVGG, self).__init__()
        
        self.args = args
        self.use_emb = False
        
        # repvgg는 원래 dropout 없음
        if 'repvgg-a0' in self.args.model:
            n_features = 1280
            
            if args.restore_path is not None:
                _model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2,
                                width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=True)
                
            else:
                _model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2,
                                width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
                
        self.feature_module = nn.Sequential(*list(_model.children())[:-1])
        self.classifier = nn.Linear(n_features, 2, bias=True)
        
        if 'hem-emb' in self.args.hem_extract_mode or 'hem-focus' in self.args.hem_extract_mode:
            self.use_emb = True
            self.proxies = nn.Parameter(torch.randn(n_features, 2))
            
        if self.args.use_online_mcd:
            self.dropout = nn.Dropout(self.args.dropout_prob)
            
    def forward(self, x):
        features = self.feature_module(x).view(x.size(0), -1)
        
        if self.args.use_online_mcd: 
            if self.training: 
                features = self.dropout(features)
            else:
                mcd_outputs = []
                for _ in range(self.args.n_dropout):
                    mcd_outputs.append(self.dropout(features).unsqueeze(0))
                    
                a = torch.vstack(mcd_outputs)
                features = torch.mean(a, 0)
            
        output = self.classifier(features)
        
        if self.use_emb and self.training:
            return features, output
        else:
            return output