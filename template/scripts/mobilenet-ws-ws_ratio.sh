# 21.11.19 general ws IB ratio별 실험
# [고정] general-ws / random_seed:3829 / top_ratio:0.10
# [변경] IB_ratio

ws_ratio_array=(2 3 4);

for ws_ratio in "${ws_ratio_array[@]}";
do
    nohup python visual_flow.py \
        --fold "1" \
        --trial 1 \
        --use_wise_sample \
        --WS_ratio ${ws_ratio} \
        --model "mobilenetv3_large_100" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 50 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "7" \
        --random_seed 3829 \
        --IB_ratio 3 \
        --hem_extract_mode "hem-softmax-offline" \
        --top_ratio 0.10 \
        --stage "general_train" \
        --inference_fold "1" \
        --experiments_sheet_dir "/OOB_RECOG/results/general_ws-ws_ratio-experiment" \
        --save_path "/OOB_RECOG/logs/general_ws-ws_ratio-experiment" > /dev/null
done;