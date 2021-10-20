from core.model.mobilenet import generate_mobilenet
from core.model.efficientnet import generate_efficientnet
from core.model.resnet import generate_resnet


name_to_model = {
    'mobilenet_v3_small': generate_mobilenet,
    'mobilenet_v3_large': generate_mobilenet,
    'efficientnet_b0': generate_efficientnet,
    'efficientnet_b1': generate_efficientnet,
    'efficientnet_b2': generate_efficientnet,
    'efficientnet_b3': generate_efficientnet,
    'efficientnet_b4': generate_efficientnet,
    'resnet18': generate_resnet,
}



def get_model(args):
    return name_to_model[args.model](args)