# 21.12.10 stage 200 apply로 뽑아버리기

top_ratio=(0.05);

WS_ratio=3;
n_dropout=1;
IB_ratio=3;

theator_stage_flag=200;

for ratio in "${top_ratio[@]}";
do
    python ../apply_offline_methods_flow.py \
        --theator_stage_flag ${theator_stage_flag}\
        --fold "1" \
        --trial 1 \
        --use_wise_sample \
        --apply_mode \
        --WS_ratio ${WS_ratio} \
        --model "resnet18" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 50 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "0" \
        --random_seed 3829 \
        --IB_ratio ${IB_ratio} \
        --hem_extract_mode "all-offline" \
        --top_ratio ${ratio} \
        --n_dropout ${n_dropout} \
        --stage "hem_train" \
        --inference_fold "1" \
        --hem_per_patient \
        --experiments_sheet_dir "/OOB_RECOG/results-${theator_stage_flag}/resnet_apply_offline_methods-all-offline-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-MC=${n_dropout}-experiment" \
        --save_path "/OOB_RECOG/logs-${theator_stage_flag}/resnet_apply_offline_methods-all-offline-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-MC=${n_dropout}-experiment"
done;