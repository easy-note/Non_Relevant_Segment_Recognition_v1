
import torch
import torchvision.models as models
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet # Add Support Model [EfficientNet Family] - https://github.com/lukemelas/EfficientNet-PyTorch

from torchsummary import summary


class CAMIO(pl.LightningModule):
    """ Define backbone model. """

    def __init__(self, config:dict):
        super(CAMIO, self).__init__()

        self.hparams.update(config) # self.hparams=config has been removed from later versions and is no longer supported. 
        self.save_hyperparameters() # save with hparams

        # hyper param setting
        self.init_lr = self.hparams.optimizer_lr # config['optimizer_lr']
        self.backbone = self.hparams.backbone_model # config['backborn_model']

        print(config)
        print(self.init_lr)
        print(self.backbone) 

        # model setting
        if self.backbone.find('mobilenet') != -1 : 
            if self.backbone == 'mobilenet_v3_large' :
                self.model = models.mobilenet_v3_large(pretrained=True)
                self.num_ftrs = self.model.classifier[-1].in_features

                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(960, self.num_ftrs), #lastconv_output_channels, last_channel
                    torch.nn.Hardswish(inplace=True),
                    torch.nn.Dropout(p=0.2, inplace=True),
                    torch.nn.Linear(self.num_ftrs, 2) #last_channel, num_classes
                )

            else :
                assert(False, '=== Not supported MobileNet model ===')


        elif self.backbone.find('efficientnet') != -1 :
            if self.backbone == 'efficientnet_b3' : 
                self.model = EfficientNet.from_pretrained('efficientnet-b3', advprop=False, num_classes=2) # Normailize from ImageNet
            
            else :
                assert(False, '=== Not supported EfficientNet model ===')            

        else :
            assert(False, '=== Not supported Model === ')
    
    def forward(self, x):
        return self.model(x)

    