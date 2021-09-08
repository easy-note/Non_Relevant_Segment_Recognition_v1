import os
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from core.config.base_opts import parse_opts
from core.api.trainer import CAMIO


def main():
    parser = parse_opts()
    args = parser.parse_args()
    print(args.save_log_path, args.dataset, args.task, args.fold)
    args.save_ckpt_path = os.path.join(args.save_log_path, args.dataset, args.task, args.fold)

    os.makedirs(args.save_ckpt_path, exist_ok=True)

    if args.train_method is not 'normal' and args.train_method is not 'hem-bs':
        args.max_epoch *= 2

    model = CAMIO(args)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save_ckpt_path,
                                            name='TB_log',
                                            default_hp_metric=False)

    if args.num_gpus > 1:
        trainer = pl.Trainer(gpus=args.num_gpus, 
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,
                            plugins=DDPPlugin(find_unused_parameters=False), # [Warning DDP] error ?
                            accelerator='ddp')
    else:
        trainer = pl.Trainer(gpus=args.num_gpus,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)

    if args.resume:
        trainer = pl.Trainer(resume_from_checkpoint=args.restore_path) 

    # trainer.fit(model)


if __name__ == '__main__':
    main()