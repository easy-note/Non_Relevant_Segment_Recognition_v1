ratio_array=(1 3 5 7 10 18)

for ratio in "${ratio_array[@]}";
do
    python visual_flow.py \
        --fold "1" \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 50 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "1" \
        --random_seed 3829 \
        --IB_ratio ${ratio} \
        --hem_extract_mode "hem-softmax-offline" \
        --stage "hem_train" \
        --inference_fold "1" \
        --experiments_sheet_dir "/OOB_RECOG/results/mobilenet-hem-offline-softmax-ratio" \
        --save_path "/OOB_RECOG/logs/mobilenet-softmax-ratio_${ratio}"
done

# 1,2,3,5,7,10,18