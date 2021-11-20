# 21.11.19 off-line method별 실험
# [고정] IB_ratio:4 / random_seed:3829 / top_ratio:10.0
# [변경] hem_extract_mode

method_array=("hem-softmax-offline" "hem-voting-offline" "hem-vi-offline");

for method in "${method_array[@]}";
do
    nohup python visual_flow.py \
        --fold "1" \
        --trial 1 \
        --wise_sampling_mode \
        --model "mobilenetv3_large_100" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 50 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "6" \
        --random_seed 3829 \
        --IB_ratio 3 \
        --hem_extract_mode ${method} \
        --top_ratio 0.10 \
        --stage "hem_train" \
        --inference_fold "1" \
        --experiments_sheet_dir "/OOB_RECOG/results/offline-methods-experiment" \
        --save_path "/OOB_RECOG/logs/offline-methods-experiment" > /dev/null
done;