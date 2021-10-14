import torch.nn as nn
import torchvision.models as models


def generate_mobilenet(args):
    arch_name = args.model

    if arch_name == 'mobilenet_v2':
        print('MODEL = MOBILENET_V2')
        model = models.mobilenet_v2(pretrained=args.pretrained)
        num_ftrs = model.classifier[-1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 2),
        )
    
    elif arch_name == 'mobilenet_v3_small' :
        print('MODEL = MOBILENET_V3_SMALL')
        # self.model = models.mobilenet_v3_small(pretrained=True)
        model = models.mobilenet_v3_small(pretrained=args.pretrained) # model scretch learning
        num_ftrs = model.classifier[-1].in_features

        model.classifier = nn.Sequential(
            nn.Linear(576, num_ftrs), #lastconv_output_channels, last_channel
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 2) #last_channel, num_classes
        )
    elif arch_name == 'mobilenet_v3_large' :
        print('MODEL = MOBILENET_V3_LARGE')
        model = models.mobilenet_v3_large(pretrained=args.pretrained)
        num_ftrs = model.classifier[-1].in_features

        model.classifier = nn.Sequential(
            nn.Linear(960, num_ftrs), #lastconv_output_channels, last_channel
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 2) #last_channel, num_classes
        )

    return model