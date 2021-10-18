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