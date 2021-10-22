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
            action='store_true', # false
            help='If true, load pretrained backbone')

    parser.add_argument('--loss_fn',
            default='ce',
            type=str,
            choices=['ce'],
            help='Select loss_fn to train/test')

    parser.add_argument('--batch_size',
            default=128,
            type=int,
            help='Training/Testing batch size')

    parser.add_argument('--min_epoch',
            default=0,
            type=int,
            help='Minimum training epoch')

    parser.add_argument('--max_epoch',
            default=20,
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
            default='1',
            type=str,
            help='Name list of gpus that are used to train')

    parser.add_argument('--use_early_stop',
            action='store_true',
            help='If true, Ealry stopping function on')

    parser.add_argument('--early_stop_monitor',
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

    # /OOB_RECOG/logs/project-1/TB_log/version_0
    parser.add_argument('--save_path', type=str, 
                        default='/OOB_RECOG/logs/211023_HEM-softmax-FOLD2', help='')

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

    parser.add_argument('--lr_milestones',
            default=[9, 14],
            type=list,
            help='Multi-step milestones for decreasing learning rate')

    # -------------- Dataset --------------------
    parser.add_argument('--dataset', 
            default='ROBOT', 
            type=str, 
            choices=['ROBOT', 'LAPA'], 
            help='[robot, lapa] choice on dataset')

    parser.add_argument('--task', 
            default='OOB', 
            type=str,
            choices=['OOB', 'NIR'], 
            help='[OOB, NIR] choice on task')

    parser.add_argument('--experiment_type', 
            default='ours', 
            type=str,
            choices=['ours', 'theator'], )

    parser.add_argument('--data_base_path',
            default='/raid/img_db',
            type=str,
            help='Data location')

    parser.add_argument('--fold',
            default='2',
            type=str,
            choices=['1', '2', '3', '4', '5', 'free'],
            help='valset 1, 2, 3, 4, 5, free=for setting train_videos, val_vidoes')

    parser.add_argument('--data_version',
            default='v3',
            type=str,
            choices=['v1', 'v2', 'v3'],
            help='Annotation dataset version')

    parser.add_argument('--IB_ratio',
            default=3,
            type=int,
            help='')

    parser.add_argument('--num_workers',
            default=6,
            type=int,
            help='How many CPUs to use for data loading')
    

    # -------------- Train Methods --------------------
    parser.add_argument('--hem-vi', type=str,
            default='hem-vi', 
            choices=['normal', 'hem-softmax', 'hem-bs', 'hem-vi'],
            help='Select train method, normal or hem method')

    parser.add_argument('--hem_bs_n_batch', type=int,
            default=4, 
            help='Set the number of batches')
    
    parser.add_argument('--top_ratio', type=float,
            default=30/100,
            help='Select HEM top ratio')

    # -------------- etc --------------------
    parser.add_argument('--random_seed', type=int, default=10, help='dataset ranbom seed')

    parser.add_argument('--use_lightning_style_save', action='store_true', help='If true, use lightning save module')

    parser.add_argument('--save_top_n', type=int, default=1, help='dataset ranbom seed')


    return parser
