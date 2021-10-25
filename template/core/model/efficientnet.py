import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def generate_efficientnet(args):
    arch_name = args.model
    arch_name = arch_name[:-3] + '-' + arch_name[-2:]

    print('MODEL = EFFICIENTNET-{}'.format(arch_name[-2:]))
    if args.pretrained:
        model = EfficientNet.from_pretrained(arch_name, advprop=False, num_classes=2)
    else:
        model = EfficientNet.from_name(arch_name, num_classes=2)

    return model


class EfficientNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.use_emb = False
    
        arch_name = self.args.model
        
        arch_name = arch_name[:-3] + '-' + arch_name[-2:]

        print('MODEL = EFFICIENTNET-{}'.format(arch_name[-2:]))
        if args.pretrained:
            model = EfficientNet.from_pretrained(arch_name, advprop=False, num_classes=2)
        else:
            model = EfficientNet.from_name(arch_name, num_classes=2)
        
        ml = list(model.children())
        
        self.feature_module = nn.Sequential(
            *ml[:-3]
        )
        self.classifier = nn.Sequential(
            *ml[-3:]
        )
        
        
    def forward(self, x):
        features = self.feature_module(x)
        output = self.classifier(features)
        
        if self.use_emb and self.training:
            return features, output
        else:
            return output