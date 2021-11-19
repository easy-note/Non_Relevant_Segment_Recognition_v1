# 21.11.19 off-line IB ratio별 실험
# [고정] hem_extract_mode:hem-voting-offline / random_seed:3829 / top_ratio:10.0
# [변경] IB_ratio

ratio_array=(3 4 5 7);

for ratio in "${ratio_array[@]}";
do
    python visual_flow.py \
        --fold "1" \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 1 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "5" \
        --random_seed 3829 \
        --IB_ratio ${ratio} \
        --hem_extract_mode "hem-voting-offline" \
        --top_ratio 10.0 \
        --stage "hem_train" \
        --inference_fold "1" \
        --experiments_sheet_dir "/OOB_RECOG/results/offline-ratio-experiment" \
        --save_path "/OOB_RECOG/logs/offline-ratio-experiment"
done;