import argparse


def parse_opts():
    """
        Base arguments parser
    """
    parser = argparse.ArgumentParser()

    # --------------- Model basic info --------------------
    parser.add_argument('--model',
            default='mobilenet_v3_large',
            type=str,
            choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 
                        'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d',
                        'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'squeezenet1_0', 'squeezenet1_1',
                        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
                        'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'],
            help='Select model to train/test')

    parser.add_argument('--pretrained',
            action='store_true',
            help='If true, load pretrained backbone')

    parser.add_argument('--loss_fn',
            default='ce',
            type=str,
            choices=['ce'],
            help='Select loss_fn to train/test')

    parser.add_argument('--batch_size',
            default=2,
            type=int,
            help='Training/Testing batch size')

    parser.add_argument('--min_epoch',
            default=0,
            type=int,
            help='Minimum training epoch')

    parser.add_argument('--max_epoch',
            default=100,
            type=int,
            help='Maximum training epoch')

    parser.add_argument('--num_gpus',
            default=1,
            type=int,
            help='How many GPUs to use for training')

    parser.add_argument('--device',
            default='cuda',
            type=str,
            help='What device to use for training or validation model')

    parser.add_argument('--cuda_list',
            default='0',
            type=str,
            help='Name list of gpus that are used to train')

    parser.add_argument('--use_early_stop',
            action='store_true',
            help='If true, Ealry stopping function on')

    parser.add_argument('--ealry_stop_monitor',
            default='val_loss',
            type=str,
            help='select monitor value')

    parser.add_argument('--ealry_stop_mode',
            default='min',
            type=str,
            help='select monitor mode')

    parser.add_argument('--ealry_stop_patience',
            default=10,
            type=int,
            help='If monitor value is not updated until this value, stop training')

    parser.add_argument('--save_log_path',
            default='./logs',
            type=str,
            help='Save training logs on this directory')

    parser.add_argument('--save_ckpt_path', type=str, 
                        default=None, help='')

    parser.add_argument('--test_mode',
            action='store_true',
            help='If true, Only testing')

    parser.add_argument('--resume',
            action='store_true',
            help='If true, keep training from the checkpoint')

    parser.add_argument('--restore_path', 
            type=str, 
            default=None, 
            help='Resume or test to train the model loaded from the path')

    # --------------- Optimizer and Scheduler info ----------------------
    parser.add_argument('--init_lr',
            default=1e-3,
            type=float,
            help='Initial learning rate')

    parser.add_argument('--weight_decay',
            default=1e-5,
            type=float,
            help='Weight decay value')

    parser.add_argument('--optimizer',
            default='adam',
            type=str,
            choices=['sgd', 'adam'],
            help=('What optimizer to use for training'
                '[Types : sgd, adam]'))

    parser.add_argument('--lr_scheduler',
            default='step_lr',
            type=str,
            choices=['step_lr', 'mul_lr'],
            help='Learning scheduler selection \n[Types : step_lr, mul_lr]')

    parser.add_argument('--lr_scheduler_step', 
            type=int, 
            default=5, 
            help='Use for Step LR Scheduler')

    parser.add_argument('--lr_scheduler_factor',
                default=0.9,
                type=float,
                help='Multiplicative factor for decreasing learning rate')

    # -------------- Dataset --------------------
    parser.add_argument('--dataset', type=str, default='ROBOT', choices=['ROBOT', 'LAPA'], help='[robot, lapa] choice on dataset')

    parser.add_argument('--task', type=str,
                        default='OOB', choices=['OOB', 'NIR'], help='[OOB, NIR] choice on task')

    parser.add_argument('--data_path',
            default=None,
            type=str,
            help='Data location')

    parser.add_argument('--fold',
            default='1',
            type=str,
            choices=['1', '2', '3', 'free'],
            help='valset 1, 2, 3, free=for setting train_videos, val_vidoes')

    parser.add_argument('--num_workers',
            default=6,
            type=int,
            help='How many CPUs to use for data loading')

    # -------------- Train Methods --------------------
    parser.add_argument('--train_method', type=str,
                        default='normal', 
                        choices=['normal', 'hem-softmax', 'hem-bs', 'hem-vi'],
                        help='Select train method, normal or hem method')


    # -------------- etc --------------------
    parser.add_argument('--random_seed', type=int, help='dataset ranbom seed')

    parser.add_argument('--use_lightinig_style_save', action='store_true', help='If true, use lightning save module')

    parser.add_argument('--save_top_n', type=int, default=3, help='dataset ranbom seed')


    return parser
    
