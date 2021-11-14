STAGE_LIST = ['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3', 'hem_train', 'general_train']

def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args

def train_main(config, args):
    print('train_main')
    
    import os
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.plugins import DDPPlugin

    from core.model import get_model, get_loss
    from core.api.trainer import CAMIO
    from core.api.theator_trainer import TheatorTrainer

    from torchsummary import summary
    import ray
    
    args.batch_size = config['batch_size'].sample()
    args.init_lr = config['lr'].sample()
    args.lr_scheduler_step = config['lr_step'].sample()
    args.lr_scheduler_factor = config['drop_ratio'].sample()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save_path,
        name='TB_log',
        default_hp_metric=False)

    if args.experiment_type == 'theator':
        x = TheatorTrainer(args)
    elif args.experiment_type == 'ours':
        x = CAMIO(args)
    
    if args.num_gpus > 1:
        trainer = pl.Trainer(gpus=args.num_gpus, 
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,
                            plugins=DDPPlugin(find_unused_parameters=False), # [Warning DDP] error ?
                            accelerator='ddp')
    else:
        trainer = pl.Trainer(gpus=args.num_gpus,
                            # limit_train_batches=2,#0.01,
                            # limit_val_batches=2,#0.01,
                            num_sanity_val_steps=0,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)
    
    trainer.fit(x)

    # args.restore_path = os.path.join(args.save_path, 'TB_log', 'version_0') # TO DO: we should define restore path
    
    args.restore_path = os.path.join(x.restore_path)
    print('restore_path: ', args.restore_path)
    
    return args

def main():    
    # 0. set each experiment args 
    import os
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    
    
    config = {
        "lr": tune.loguniform(1e-3, 1e-1),
        # "lr": tune.uniform(1e-3, 1e-1),
        # "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
        # "lr": tune.choice([1e-1, 1e-2, 1e-3, 2e-2, 5e-3, 7e-2]),
        "lr_step": tune.uniform(1,10),
        "drop_ratio": tune.loguniform(0.1, 0.9),
        "batch_size": tune.choice([32, 64, 128, 256]),
    }
    
    num_epochs = 10
    gpus_per_trial = 1
    data_dir = './ray-results'
    num_samples = 10
    
    scheduler = ASHAScheduler(max_t=num_epochs,
                                grace_period=1,
                                reduction_factor=2)

    reporter = CLIReporter(metric_columns=["Loss", "mean_metric", "training_iteration"])
    
    # 0. set each experiment args 
    args = get_experiment_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    # general mode
    if args.stage == 'general_train':
        args.mini_fold = 'general'
        args = train_main(config, args)
        
    else: # online mode
        if check_hem_online_mode(args):
            args.mini_fold = 'general'
            args.stage = 'hem_train'

            args = train_main(args)
            
    analysis = tune.run(
        tune.with_parameters(
            train_main,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            data_dir=data_dir),
            resources_per_trial={
                "cpu": 4,
                "gpu": gpus_per_trial
            },
            metric="Loss",
            mode="min",
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="tune_mnist_asha"
    )

    print("Best hyperparameters found were: ", analysis.best_config)
            
            
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        print(base_path)
        
        from core.utils.misc import save_dict_to_csv, prepare_inference_aseets, get_inference_model_path, \
    set_args_per_stage, check_hem_online_mode, clean_paging_chache

    main()



# python ray-test.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
# --lr_scheduler "step_lr" --cuda_list '6' \
# --IB_ratio 3 --random_seed 10 --stage 'general_train' --inference_fold '1' \
# --emb_type 3 --save_path '/code/OOB_Recog/logs/general-ray-test'
