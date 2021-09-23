def main():
    print('main')
    
    ### test logging module
    from core.config.base_opts import parse_opts
    from core.utils.logging import LogHelper # Logging module

    import os

    parser = parse_opts()

    # -------------- Inference Methods --------------------
    parser.add_argument('--inference_save_dir', type=str, 
                        default='../../restuls',
                        help='root directory for infernce saving')
        

    parser.add_argument('--step_of_inference', type=int, 
                        default=30,
                        help='Inference frame step of Evaluation')

    parser.add_argument('--inference_fold',
                    default='3',
                    type=str,
                    choices=['1', '2', '3', '4', '5', 'free'],
                    help='valset 1, 2, 3, free=for setting train_videos, val_vidoes')

    args = parser.parse_args()

    # inference logging
    train_model_info = '{}-{}-{}-{}-{}'.format(args.model, args.dataset, args.task, args.fold, args.train_method)
    inference_info = 'inference_fold-{}'.format(args.inference_fold)

    inference_save_path = os.path.join(args.inference_save_dir, train_model_info, inference_info)
    os.makedirs(inference_save_path, exist_ok=True)

    log_helper = LogHelper(os.path.join(inference_save_path, 'log.txt')) # logging
    log_helper.writeln('\t === START INFERENCE === \t\n')
    log_helper.writeln()


    # INFERENCEING ... 


    log_helper.writeln('\t === END INFERENCE === \t\n')
    log_helper.writeln()

if __name__ == '__main__':
    
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    main()