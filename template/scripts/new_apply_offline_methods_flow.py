def get_experiment_args():
    
    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    # args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    # args.save_path = args.save_path + '-model:{}-IB_ratio:{}-WS_ratio:{}-hem_extract_mode:{}-top_ratio:{}-seed:{}'.format(args.model, args.IB_ratio, args.WS_ratio, args.hem_extract_mode, args.top_ratio, args.random_seed) # offline method별 top_ratio별 IB_ratio별 실험을 위해
    # args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args

def get_clean_args():


    parser = parse_opts()
    args = parser.parse_args()
    
    cleaned_args = argparse.Namespace()

    for key, value in args._get_kwargs(): 
        setattr(cleaned_args, key, None)

    return cleaned_args

def get_extract_hem_assets_args(args):

    # 1. clean
    extract_hem_assets_args = get_clean_args()

    setting_dict =  {
        'model': args.model, # var
        'n_dropout': args.n_dropout, # var
        'experiment_type': 'ours',
        'IB_ratio': 3,
        'WS_ratio': 3,
        'top_ratio': args.top_ratio, # var
        'random_seed': 3829,
        'hem_extract_mode': None,
        'use_wise_sample': True,
        'dropout_prob': 0.3,
        'loss_fn': 'ce',
        'max_epoch': 0,
        'use_online_mcd': False,
        'hem_interation_idx': args.hem_interation_idx, # var
        'baby_model_save_path': args.baby_model_save_path, # var
        'experiment_sub_type': 'none', # only in semi (dataset)
        'semi_data': 'rs-general', # only in semi (dataset)
    }

    # 2. only set for extract hem assets from input args
    for key, value in setting_dict.items(): 
        setattr(extract_hem_assets_args, key, value)

    return extract_hem_assets_args

def train_main(args):
    print('train_main')
    


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
        if args.use_test_batch:
            trainer = pl.Trainer(gpus=args.num_gpus,
                            limit_train_batches=2,
                            limit_val_batches=2,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)
        else:    
            trainer = pl.Trainer(gpus=args.num_gpus,
                            # limit_train_batches=2,#0.01,
                            # limit_val_batches=2,#0.01,
                            max_epochs=args.max_epoch, 
                            min_epochs=args.min_epoch,
                            logger=tb_logger,)
    
    trainer.fit(x)

    # args.restore_path = os.path.join(args.save_path, 'TB_log', 'version_0') # TO DO: we should define restore path
    
    args.restore_path = os.path.join(x.restore_path)
    print('restore_path: ', args.restore_path)
    
    return args

def inference_main(args):
    print('inference_main')
    


    # from pretrained model
    '''
    model = CAMIO(args)
    model = model.cuda()
    '''

    print('restore : ', args.restore_path)

    # from finetuning model
    model_path = get_inference_model_path(os.path.join(args.restore_path, 'checkpoints'))
    model = CAMIO.load_from_checkpoint(model_path, args=args) # .ckpt
        
    model = model.cuda()

    # inference block
    os.makedirs(args.restore_path, exist_ok=True)

    inference_assets_save_path = args.restore_path
    details_results_path = os.path.join(args.restore_path, 'inference_results')
    patients_results_path = os.path.join(details_results_path, 'patients_report.csv')
    videos_results_path = os.path.join(details_results_path, 'videos_report.csv')

    report_path = os.path.join(args.restore_path, 'Report.json')


    # 1. load inference dataset
    # inference case(ROBOT, LAPA), anno_ver(1,2,3), inference fold(1,2,3,4,5)에 따라 환자별 db. gt_json 잡을 수 있도록 set up
    case = args.dataset # ['ROBOT', LAPA]
    anno_ver = '3'
    inference_fold = args.inference_fold

    inference_assets = prepare_inference_aseets(case=case , anno_ver=anno_ver, inference_fold=inference_fold, save_path=inference_assets_save_path)
    patients = inference_assets['patients']
    
    patients_count = len(patients)

    # 2. for record metrics
    # Report
    report = Report(report_path)

    patients_metrics_list = [] # save each patients metrics
    for idx in range(patients_count): # per patients
        patient = patients[idx]
        patient_no = patient['patient_no']
        patient_video = patient['patient_video']

        videos_metrics_list = [] # save each videos metrics
        
        patient_gt_list = [] # each patients gt list for visual
        patient_predict_list = [] # each patients predict list for visual

        # for save patients results
        each_patients_save_dir = os.path.join(details_results_path, patient_no)
        os.makedirs(each_patients_save_dir, exist_ok=True)

        for video_path_info in patient['path_info']: # per videos
            video_name = video_path_info['video_name']
            video_path = video_path_info['video_path']
            annotation_path = video_path_info['annotation_path']
            db_path = video_path_info['db_path']

            # Inference module
            inference = InferenceDB(args, model, db_path, args.inference_interval) # Inference object // args => InferenceDB init의 DBDataset(args) 생성시 args.model로 'mobile_vit' augmentation 처리해주기 위해
            predict_list, target_img_list, target_frame_idx_list = inference.start() # call start
  
            # for save video results
            each_videos_save_dir = os.path.join(each_patients_save_dir, video_name)
            os.makedirs(each_videos_save_dir, exist_ok=True)

            # save predict list to csv
            predict_csv_path = os.path.join(each_videos_save_dir, '{}.csv'.format(video_name))
            predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'target_img': target_img_list,
                        })
            predict_df.to_csv(predict_csv_path)

            # Evaluation module
            evaluator = Evaluator(predict_csv_path, annotation_path, args.inference_interval)
            gt_list, predict_list = evaluator.get_assets() # get gt_list, predict_list by inference_interval

            # save predict list to csv
            predict_csv_path = os.path.join(each_videos_save_dir, '{}.csv'.format(video_name))
            predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'gt': gt_list,
                            'target_img': target_img_list,
                        })
            predict_df.to_csv(predict_csv_path)

            # for visulization per patients
            patient_gt_list += gt_list
            patient_predict_list += predict_list

            # metric per video
            video_metrics = evaluator.calc() # same return with metricHelper
            video_CR, video_OR = video_metrics['CR'], video_metrics['OR']
            video_TP, video_FP, video_TN, video_FN = video_metrics['TP'], video_metrics['FP'], video_metrics['TN'], video_metrics['FN']
            video_TOTAL = video_FP + video_TP + video_FN + video_TN

            video_gt_IB, video_gt_OOB, video_predict_IB, video_predict_OOB = video_metrics['gt_IB'], video_metrics['gt_OOB'], video_metrics['predict_IB'], video_metrics['predict_OOB']

            video_jaccard = video_metrics['Jaccard']

            video_precision, video_recall = video_metrics['Precision'], video_metrics['Recall']

            print('\t => video_name: {}'.format(video_name))
            print('\t    video_CR: {:.3f} | video_OR: {:.3f}'.format(video_CR, video_OR))
            print('\t    video_TP: {} | video_FP: {} | video_TN: {} | video_FN: {}'.format(video_TP, video_FP, video_TN, video_FN))

            # save video metrics
            video_results_dict = report.add_videos_report(patient_no=patient_no, video_no=video_name, FP=video_FP, TP=video_TP, FN=video_FN, TN=video_TN, TOTAL=video_TOTAL, CR=video_CR, OR=video_OR, gt_IB=video_gt_IB, gt_OOB=video_gt_OOB, predict_IB=video_predict_IB, predict_OOB=video_predict_OOB, precision=video_precision, recall=video_recall, jaccard=video_jaccard)
            save_dict_to_csv(video_results_dict, videos_results_path)

            # for calc patients metric
            videos_metrics_list.append(video_metrics)
        
        # calc each patients CR, OR
        patient_metrics = MetricHelper().aggregate_calc_metric(videos_metrics_list)
        patient_CR, patient_OR = patient_metrics['CR'], patient_metrics['OR']
        patient_TP, patient_FP, patient_TN, patient_FN = patient_metrics['TP'], patient_metrics['FP'], patient_metrics['TN'], patient_metrics['FN']
        patient_TOTAL = patient_FP + patient_TP + patient_FN + patient_TN

        patient_gt_IB, patient_gt_OOB, patient_predict_IB, patient_predict_OOB = patient_metrics['gt_IB'], patient_metrics['gt_OOB'], patient_metrics['predict_IB'], patient_metrics['predict_OOB']

        patient_jaccard = patient_metrics['Jaccard']

        patient_precision, patient_recall = patient_metrics['Precision'], patient_metrics['Recall']

        print('\t\t => patient_no: {}'.format(patient_no))
        print('\t\t    patient_CR: {:.3f} | patient_OR: {:.3f}'.format(patient_CR, patient_OR))
        print('\t\t    patient_TP: {} | patient_FP: {} | patient_TN: {} | patient_FN: {}'.format(patient_TP, patient_FP, patient_TN, patient_FN))

        # save patient metrics        
        patient_results_dict = report.add_patients_report(patient_no=patient_no, FP=patient_FP, TP=patient_TP, FN=patient_FN, TN=patient_TN, TOTAL=patient_TOTAL, CR=patient_CR, OR=patient_OR, gt_IB=patient_gt_IB, gt_OOB=patient_gt_OOB, predict_IB=patient_predict_IB, predict_OOB=patient_predict_OOB, precision=patient_precision, recall=patient_recall, jaccard=patient_jaccard)
        save_dict_to_csv(patient_results_dict, patients_results_path)
    
        # for calc total patients CR, OR
        patients_metrics_list.append(patient_metrics)
        
        # visualization per patients
        patient_predict_visual_path = os.path.join(each_patients_save_dir, 'predict-{}.png'.format(patient_no))

        visual_tool = VisualTool(patient_gt_list, patient_no, patient_predict_visual_path)
        visual_tool.visual_predict(patient_predict_list, args.model, args.inference_interval, window_size=300, section_num=2)

        # CLEAR PAGING CACHE
        # clean_paging_chache()

    # for calc total patients CR, OR + (mCR, mOR)
    total_metrics = MetricHelper().aggregate_calc_metric(patients_metrics_list)
    total_mCR, total_mOR, total_CR, total_OR = total_metrics['mCR'], total_metrics['mOR'], total_metrics['CR'], total_metrics['OR']
    total_mPrecision, total_mRecall = total_metrics['mPrecision'], total_metrics['mRecall']
    total_Jaccard = total_metrics['Jaccard']

    report.set_experiment(model=args.model, methods=args.hem_extract_mode, inference_fold=args.inference_fold, mCR=total_mCR, mOR=total_mOR, CR=total_CR, OR=total_OR, mPrecision=total_mPrecision, mRecall=total_mRecall, Jaccard=total_Jaccard, details_path=details_results_path, model_path=model_path)
    report.save_report() # save report

    # SUMMARY
    patients_CR = report.get_patients_CR()
    patients_OR = report.get_patients_CR()

    experiment_summary = {
        'model':args.model,
        'methods':args.hem_extract_mode,
        'top_ratio':args.top_ratio,
        'stage': args.train_stage,

        'random_seed': args.random_seed,
        'IB_ratio': args.IB_ratio,
        'WS_ratio': args.WS_ratio,
        'inference_fold':args.inference_fold,

        'mCR':total_mCR,
        'mOR':total_mOR,
        'CR':total_CR,
        'OR':total_OR,
        'mPrecision':total_mPrecision,
        'mRecall': total_mRecall,
        'Jaccard': total_Jaccard,

        'details_path':details_results_path,
        'model_path': model_path,
    }
    
    # return mCR, mOR, OR, CR of experiment
    return args, experiment_summary, patients_CR, patients_OR


def get_baby_model_path_from_NAS(hem_interation_idx, model_name, train_stage):
    

    # train_stage의 경우 args 상관없이 미리 정의된 extract_stage에서 for문 돌면서 셋팅된 값 받아서 해당 함수로 call
    # model 과 hem_iter 만 변경 (전체 flow에서는 extract_args에서 셋팅된 값 받아서 해당 함수로 call)
    # hem_interation_idx = 100
    # model_name = 'mobilenetv3_large_100'
    
    # 무조건 고정항목
    WS_ratio = 3
    IB_ratio = 3
    random_seed = 3829

    model_dir = os.path.join(mc_assets_save_path['robot'], 'models', 'theator_stage_flag={}'.format(hem_interation_idx), model_name, 'WS={}-IB={}-seed={}'.format(WS_ratio, IB_ratio, random_seed), train_stage)
    
    return model_dir

def get_baby_model_path_from_restore_dir(baby_model_save_path, train_stage):
    

    # visual flow에서 학습한 baby model dir을 넣어주어서 train_stage에 따라서 model path 가져오기
    train_stage_to_restore_path = {
        'mini_fold_stage_0': 'TB_log/version_0/checkpoints',
        'mini_fold_stage_1': 'TB_log/version_1/checkpoints',
        'mini_fold_stage_2': 'TB_log/version_2/checkpoints',
        'mini_fold_stage_3': 'TB_log/version_3/checkpoints',
    }

    model_dir = os.path.join(baby_model_save_path, train_stage_to_restore_path[train_stage])
    
    return model_dir

def save_hem_assets_info(hem_assets_df, save_path):
    

    NON_HEM, HEM = (0, 1)
    RS_CLASS, NRS_CLASS = (0, 1)

    vanila_neg_df = hem_assets_df[(hem_assets_df['class_idx'] == RS_CLASS) & (hem_assets_df['HEM'] == NON_HEM)]
    vanila_pos_df = hem_assets_df[(hem_assets_df['class_idx'] == NRS_CLASS) & (hem_assets_df['HEM'] == NON_HEM)]
    hard_neg_df = hem_assets_df[(hem_assets_df['class_idx'] == RS_CLASS) & (hem_assets_df['HEM']  == HEM)]
    hard_pos_df = hem_assets_df[(hem_assets_df['class_idx'] == NRS_CLASS) & (hem_assets_df['HEM'] == HEM)]

    save_data = {
        'rs': {
            'hem': len(hard_neg_df),
            'vanila': len(vanila_neg_df),
        },
        'nrs': {
            'hem': len(hard_pos_df),
            'vanila': len(vanila_pos_df),
        }
    }
    
    print('save data', save_data)

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    return save_path


def extract_hem_assets(extract_args, offline_methods, save_path): # save_path 는 저장하는 ~/TB_log/version_0 에서의 ~
    
    hem_assets_paths = {} # return 

    extract_stage = ['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3']

    logger_path = os.path.join(save_path, 'TB_log') # tb_logger path

    train_stage_to_restore_path = {
        'mini_fold_stage_0':os.path.join(logger_path, 'version_0'),
        'mini_fold_stage_1':os.path.join(logger_path, 'version_1'),
        'mini_fold_stage_2':os.path.join(logger_path, 'version_2'),
        'mini_fold_stage_3':os.path.join(logger_path, 'version_3'),
    }

    train_stage_to_minifold = {
                        'mini_fold_stage_0': '1',
                        'mini_fold_stage_1': '2',
                        'mini_fold_stage_2': '3',
                        'mini_fold_stage_3': '4',
                    }
    
    for train_stage in extract_stage:

        restore_path = train_stage_to_restore_path[train_stage] # set restore_path
        os.makedirs(restore_path, exist_ok=True) # for saveing version 0,1,2,3

        # 1. baby model (extract model) 불러오기
        if extract_args.baby_model_save_path == '': # 1-1. model 불러오기 (from NAS)
            model_dir = get_baby_model_path_from_NAS(extract_args.hem_interation_idx, extract_args.model, train_stage)
        else : # 1-2. local에서 불러오기
            model_dir = get_baby_model_path_from_restore_dir(extract_args.baby_model_save_path, train_stage)
        
        print(model_dir)

        # args는 model 불러올떄만 사용하기 위해(checkpoint)
        # args = get_experiment_args()
        
        # ------------ 1. timm model 불러올때 사용하는 args ------------ #
        args = get_clean_args()
        # ==> 공용 
        # args.model = model_name
        args.model = extract_args.model # 'mobilenetv3_large_100' # model 정보만 정확하게 넘겨주면 되지 않을까?????
        
        args.experiment_type = 'ours'
        args.hem_extract_mode = 'offline-multi' # 어차피 사용안됨, 그냥 넣어줌
        args.dropout_prob = 0.3
        args.loss_fn = 'ce' # get_loss function

        # ==> offline 
        args.max_epoch = 50 # trainer function (init에 존재)

        # ==> online // model 불러올때..
        args.use_online_mcd = False
        args.n_dropout = 1 
        # ------------ ------------- ------------ #

        
        model_path = get_inference_model_path(model_dir)    
        model = CAMIO.load_from_checkpoint(model_path, args=args) # .ckpt
        '''
        pt_path=None # for using change_deploy_mode for offline, it will be update on above if's branch

        if 'repvgg' in args.model: 
            pt_path = get_pt_path(model_dir)
            print('\n\t ===> LOAD PT FROM {}\n'.format(pt_path))
        
        model.change_deploy_mode(pt_path=pt_path) # change feature_module weight from saved pt
        '''

        model = model.cuda()
        # model.eval() # 어차피 mc dropout 에서 처리
        # 1-2. train/validation set 불러오기 // train set 불러오는 이유는 hem extract 할때 얼마나 뽑을지 정해주는 DATASET_COUNT.json을 저장하기 위해
        # RobotDataset 의 경우 내부적으로 가장 먼저 args.mini_fold 가 'general'인가 아닌가를 기준으로 hem dataset(20/60)과 아닌걸(80)으로 나눔

        # ------------ 2.  dataset 불러올때 사용하는 args ------------ # (train)
        args = get_clean_args()
        # ==> 공용 
        args.experiment_type = 'ours'
        args.model = extract_args.model
        args.IB_ratio = 3  # hueristic sampler 에서도 사용
        args.WS_ratio = 3 # hueristic sampler 에서 사용
        args.random_seed = 3829
        args.fold = '1'
        args.use_wise_sample = True # 사실 이건 mini fold stage에서 wise sampling 햇냐 안햇냐의 재현 여부
        # ==> semi
        args.experiment_sub_type = extract_args.experiment_sub_type # 'semi' or 'none'
        args.semi_data == extract_args.semi_data # 'rs-general'
        # ------------ ------------- ------------ #

        trainset = RobotDataset_new(args, state='train_mini', minifold=train_stage_to_minifold[train_stage], wise_sample=args.use_wise_sample) # train dataset setting

        train_dataset_info_path = os.path.join(restore_path, 'train_reproduce_dataset_info.json')
        save_dataset_info(trainset, train_dataset_info_path)

        # ------------ 3.  dataset 불러올때 사용하는 args ------------ # (val)
        # 사실 validation set불러올때는 metset, all sample 사용하기 때문에 args 관련 부분 사용x 근데 안돌아가니까 사용
        args = get_clean_args()
        # ==> 공용 
        args.experiment_type = 'ours'
        args.model = extract_args.model
        args.IB_ratio = 3  # hueristic sampler 에서도 사용
        args.WS_ratio = 3 # hueristic sampler 에서 사용
        args.random_seed = 3829
        args.fold = '1'
        # ==> semi
        args.experiment_sub_type = extract_args.experiment_sub_type # 'semi' or 'none'
        args.semi_data == extract_args.semi_data # 'rs-general'
        # ------------ ----save_hem_assets_info--------- ------------ #

        val_all_metaset = RobotDataset_new(args, state='val_mini',  minifold=train_stage_to_minifold[train_stage], all_sample=True, use_metaset=True) # val dataset setting

        val_dataset_info_path = os.path.join(restore_path, 'val_all_metaset_info.json')
        save_dataset_info(val_all_metaset, val_dataset_info_path)


        # 2. hem_methods 적용
        
        # ------------ 4.  hem_helper args ------------ # 
        args = get_clean_args()

        n_dropout = extract_args.n_dropout  # hem_helper_init
        hem_extract_mode = 'offline-multi' # hem_helper_init
        use_hem_per_patient = True # hem_helper init 
        
        IB_ratio = 3 # 이건 get_target_patient_hem_count
        random_seed = 3829 # 이건 set_ratio에서 사용

        top_ratio = extract_args.top_ratio

        hem_helper = HEMHelper(args)
        hem_helper.set_method(hem_extract_mode)
        hem_helper.set_offline_multi_stack(offline_methods) # 한번에 처리
        hem_helper.set_restore_path(restore_path)
        hem_helper.set_n_dropout(n_dropout)
        hem_helper.set_use_hem_per_patient(use_hem_per_patient)
        hem_helper.set_IB_ratio(IB_ratio)
        hem_helper.set_random_seed(random_seed)
        hem_helper.set_top_ratio(top_ratio)

        ### => dataset_info load and set target hem cnt
        f_loader = FileLoader()
        f_loader.set_file_path(train_dataset_info_path)
        train_dataset_info = f_loader.load()

        f_loader.set_file_path(val_dataset_info_path)
        val_dataset_info = f_loader.load()
        
        hem_helper.set_target_hem_count(train_dataset_info['target_hem_count']['rs'], train_dataset_info['target_hem_count']['nrs'])
        hem_helper.set_target_patient_dict(val_dataset_info['patients'])

        results_dict = hem_helper.compute_hem(model, val_all_metaset) # {'hem-softmax_diff_small-offline(1)':saved_save_path(1), 'hem-softmax_diff_large-offline(2)':saved_save_path(2), ..}

        # 3. append hem csv path from results dict(return compute)
        for method_idx, method in enumerate(offline_methods, 1):
            method_info = '{}({})'.format(method, method_idx)

            if method_info in hem_assets_paths: # 같은 method 를 여러 번 수행했을 때. 
                hem_assets_paths[method_info].append(results_dict[method_info])
            else:
                hem_assets_paths[method_info] = [results_dict[method_info]]
    
    
    # 4. aggregation hem path
    all_hem_assets_dir = os.path.join(save_path, 'hem_assets')
    os.makedirs(all_hem_assets_dir, exist_ok=True)
    
    for method_info, paths in hem_assets_paths.items():
        agg_hem_assets_path = os.path.join(all_hem_assets_dir, '{}-agg.csv'.format(method_info))
        paths = natsort.natsorted(paths)

        agg_hem_assets_df = pd.DataFrame([]) # init
        for path in paths:
            hem_df = pd.read_csv(path)
            print(hem_df)
            agg_hem_assets_df = agg_hem_assets_df.append(hem_df, ignore_index=True)

        agg_hem_assets_df.to_csv(agg_hem_assets_path)
        save_hem_assets_info(agg_hem_assets_df, os.path.join(all_hem_assets_dir, '{}-agg.json'.format(method_info)))

        hem_assets_paths[method_info] = agg_hem_assets_path # re-define hem_assets_paths to agg hem assets path

    
    return hem_assets_paths


def apply_offline_methods_main(args, apply_offline_methods):
    
    # 0. mini fold args 통합 및 정리 (def extract_hem_assets 에서 사용될 독립 args)
    extract_hem_assets_args = get_extract_hem_assets_args(args)
    
    hem_assets_paths = extract_hem_assets(extract_args=extract_hem_assets_args, offline_methods=apply_offline_methods, save_path=args.save_path)

    # print(hem_assets_paths)
    print('----\n\n')
    print(hem_assets_paths)    

    # 2. train from hem assets
    for method_idx, method in enumerate(apply_offline_methods, 1):
        method_info = '{}({})'.format(method, method_idx)
        hem_assets_path = hem_assets_paths[method_info]

        ####### hem assets visualization #######
        
        try:
            hem_df = pd.read_csv(hem_assets_path)
            model = args.model
            save_dir = os.path.join('/'.join(hem_assets_path.split('/')[:-1]), method_info)

            test_visual_sampling.visual_flow_for_sampling(hem_df, model, save_dir)

        except:
            print('visual error')
            pass
        


        args.train_stage = 'hem_train'
        args.appointment_assets_path = hem_assets_path
        args.hem_extract_mode = method

        args = train_main(args) # for iter 돌면서 args.restore_path가 이전 정보로 trainer init 될 것, 하지만 sanity check 이후 다시 restore_path 재설정하므로 해당 for문 괜찮을듯.

        # 3. inference
        args, experiment_summary, patients_CR, patients_OR = inference_main(args)

        # 4. save experiments summary
        experiments_sheet_path = os.path.join(args.experiments_sheet_dir, 'experiments_summary-fold_{}.csv'.format(args.inference_fold))
        os.makedirs(args.experiments_sheet_dir, exist_ok=True)

        save_dict_to_csv({**experiment_summary, **patients_CR}, experiments_sheet_path)

def main():
    # 0. set each experiment args 
    args = get_experiment_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"]=str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

    if args.hem_extract_mode == 'offline-multi':
        apply_offline_methods = ['hem-softmax_diff_small-offline', 'hem-softmax_diff_large-offline', 'hem-voting-offline', 'hem-mi_small-offline', 'hem-mi_large-offline']
    else:
        apply_offline_methods = [args.hem_extract_mode]
    
    apply_offline_methods_main(args, apply_offline_methods=apply_offline_methods)
    

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core/accessory/RepVGG')
        print(base_path)
        
        from core.utils.misc import prepare_inference_aseets, get_inference_model_path, \
            clean_paging_chache, save_dict_to_csv, save_dataset_info

        ### get args
        import argparse
        from core.config.base_opts import parse_opts        

        ### train main
        import os
        import pytorch_lightning as pl
        from pytorch_lightning import loggers as pl_loggers
        from pytorch_lightning.plugins import DDPPlugin

        from core.model import get_model, get_loss
        from core.api.trainer import CAMIO
        from core.api.theator_trainer import TheatorTrainer

        from torchsummary import summary

        from scripts.unit_test import test_visual_sampling

        ### test inference module
        # from core.api.trainer import CAMIO
        # from core.api.theator_trainer import TheatorTrainer
        from core.api.inference import InferenceDB # inference module
        from core.api.evaluation import Evaluator # evaluation module
        from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)
        from core.utils.logger import Report # report helper (for experiments reuslts and inference results)

        from core.api.visualization import VisualTool # visual module

        ### extract hem assets
        # from core.api.trainer import CAMIO
        from core.dataset.robot_dataset_new import RobotDataset_new
        from core.config.assets_info import mc_assets_save_path
        from core.dataset.hem_methods import HEMHelper

        ### etc
        from core.utils.parser import FileLoader

        import os, torch, random
        import numpy as np
        import pandas as pd
        import glob
        import json
        import natsort


    
    main()