def main():
    print('main')
    
    ### test inference module
    from core.api.trainer import CAMIO
    from core.config.base_opts import parse_opts
    from core.api.inference import InferenceDB # inference module

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

    # from pretrained model
    model = CAMIO(args)
    model = model.cuda()

    # from finetuning model
    '''
    model_path = '/OOB_RECOG/model_ckpt/ckpoint_0816-test-mobilenet_v3_large-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt'
    model = CAMIO.load_from_checkpoint(model_path, args=args)
    '''

    DB_path = '/data2/Public/OOB_Recog/img_db/ROBOT/R_100/01_G_01_R_100_ch1_01' # R_100_ch1_01

    Inference = InferenceDB(model, DB_path) # Inference object
    Inference.set_step_of_inference(args.step_of_inference) # set inference step
    Inference.set_batch_size(args.batch_size) # set batch
    predict_list = Inference.start() # call start

    # Inference module results
    print(len(predict_list))
    print(predict_list)

if __name__ == '__main__':
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    
    if __package__ is None:
        import sys
        from os import path    
        sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    main()