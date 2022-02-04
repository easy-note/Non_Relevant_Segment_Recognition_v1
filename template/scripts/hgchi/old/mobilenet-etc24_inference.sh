# 21.12.22 etc 24 (stage100)
# mobilenet, stage 별로 가장 좋았던 methods로 학습한 Model (ws, ib ratio 3 mc = 5 random=3829 softmax diff lower로 고정, stage1:top_ratio=0.05, stage2:top_ratio=0.05)

model='mobilenetv3_large_100'
top_ratio=0.05;
WS_ratio=3;
n_dropout=5;
IB_ratio=3;

nohup python ../inference_only_etc.py \
    --fold "1" \
    --use_wise_sample \
    --WS_ratio ${WS_ratio} \
    --model ${model} \
    --pretrained \
    --use_lightning_style_save \
    --max_epoch 100 \
    --batch_size 256 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --cuda_list "0" \
    --random_seed 3829 \
    --IB_ratio ${IB_ratio} \
    --hem_extract_mode "hem-softmax-offline" \
    --top_ratio ${top_ratio} \
    --n_dropout ${n_dropout} \
    --stage "hem_train" \
    --inference_fold "all" \
    --hem_per_patient \
    --experiments_sheet_dir "/OOB_RECOG/results/etc24-${model}-inference" \
    --save_path "/OOB_RECOG/logs/${model}-theator_stage100-offline-sota/version_4" > /dev/null