# 22-01.18 theator stage 200 apply로 뽑아서 hem train시키기

# hem_train 돌릴때 기준으로 생각해서 셋팅하면 됨
# apply에서는 hem_extract_mode 일단은 "offine"으로 해놓기, 사실 뭐든 offline중 아무거나 해놓으면 됨 => 현재는 내부에 작성된 []에서 hem_extract_mode가 초기화되기 때문에
# apply면 train_stage 알아서 setup하기 때문에 아무거나 해도 되긴하는데, 의미상 hem_train으로 해놓기
# extract stage (mini_fold) 에서 nas에 저장된 model 불러올때 scripts로 넘긴 args중 theator_stage_flag, model 두가지만 영향줄 것, (WS_ratio, IB_ratio, random_seed 고정 값 + train_stage(mini_fold0,1,2,3)도 알아서 셋팅)
# use_wise_sample은 mini_fold stage에서 wise sampling 했냐 안했냐의 재현성?을 위해 사실 저장된 baby 모델은 모두 wise sampling이긴함 => 이게 baby model trainset개수셀떄 형향이 있을 꺼임.. 
# WS_ratio, IB_ratio도 재현성 문제, 어차피 baby model은 WS_ratio 3, IB_ratio 3 으로 학습한 모델이라 사용 scripts args로 설정해줘도 적용안될 것
# --hem_per_patinets 이거는 base_opts에서 제거해버렸음, hem_helper를 setter func 형식으로 바꾸었기 때문에 apply에서는 True로 고정적으로 넣어버렸음. (더이상 args로 control하기 않기)
# appointment_assets_path는 args긴 한데, apply scripts 에서 사용하는 args는 아님, 내부적으로 robot dataset불러올때 내부 로직에서 알아서 setting 해주는 용도라 scripts에서 넘기지 않음

top_ratio=(0.05);

WS_ratio=3;
n_dropout=5;
IB_ratio=3;

hem_interation_idx=100;

for ratio in "${top_ratio[@]}";
do
    nohup python ../new_apply_offline_methods_flow.py \
        --hem_interation_idx ${hem_interation_idx}\
        --fold "1" \
        --use_wise_sample \
        --WS_ratio ${WS_ratio} \
        --model "mobilenetv3_large_100" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "1" \
        --random_seed 3829 \
        --IB_ratio ${IB_ratio} \
        --hem_extract_mode "offline" \
        --top_ratio ${ratio} \
        --n_dropout ${n_dropout} \
        --train_stage "hem_train" \
        --inference_fold "1" \
        --inference_interval "30" \
        --experiments_sheet_dir "/OOB_RECOG/results/new-test_apply-mobilenet-vanila3" \
        --save_path "/OOB_RECOG/logs-new-test_apply-mobilenet-vanila3" > "./apply3.out"
done;