# 21.12.15 off-line stage 300 version 학습


IB_ratio=3;

top_ratio=(0.05);
WS_ratio=3;
n_dropout=5;

theator_stage_flag=300;

for ratio in "${top_ratio[@]}";
do
    python ../visual_flow.py \
        --theator_stage_flag ${theator_stage_flag} \
        --fold "1" \
        --trial 1 \
        --use_wise_sample \
        --WS_ratio ${WS_ratio} \
        --model "mobilenetv3_large_100" \
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
        --inference_fold "1" \
        --top_ratio ${ratio} \
        --n_dropout ${n_dropout} \
        --stage "hem_train" \
        --hem_per_patient \
        --hem_extract_mode "hem-softmax-offline" \
        --experiments_sheet_dir "/OOB_RECOG/results-${theator_stage_flag}/mobilenet_stage_softmax-diff-lower-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-top_ratio=${top_ratio}-MC=${n_dropout}-experiment" \
        --save_path "/OOB_RECOG/logs-${theator_stage_flag}/mobilenet_stage_softmax-diff-lower-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-top_ratio=${top_ratio}-MC=${n_dropout}-experiment"
done;