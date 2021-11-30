# 21.11.26 off-line mc assets 뽑기
# [고정] hem_extract_mode:all-offline / random_seed:3829 / IB_ratio:3 / WS_ratio:2 / mc:1

top_ratio=(0.50);

for ratio in "${top_ratio[@]}";
do
    nohup python visual_flow.py \
        --fold "1" \
        --trial 1 \
        --use_wise_sample \
        --WS_ratio 2 \
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
        --hem_extract_mode "all-offline" \
        --top_ratio ${ratio} \
        --n_dropout 1 \
        --stage "hem_train" \
        --inference_fold "1" \
        --experiments_sheet_dir "/OOB_RECOG/results/all-offline-MC=1-ws_ratio=2-experiment" \
        --save_path "/OOB_RECOG/logs/all-offline-MC=1-ws_ratio=2-experiment" > /dev/null
done;