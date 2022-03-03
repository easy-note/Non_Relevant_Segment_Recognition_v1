# 03.03 feature/develop test용 (MLOps example)

IB_ratio=3;

hem_interation_idx=100;

nohup python -u ../new_visual_flow.py \
    --use_test_batch \
    --hem_interation_idx ${hem_interation_idx}\
    --fold "1" \
    --model "mobilenetv3_large_100" \
    --pretrained \
    --use_lightning_style_save \
    --max_epoch 2 \
    --batch_size 256 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --cuda_list "5" \
    --random_seed 3829 \
    --IB_ratio ${IB_ratio} \
    --train_stage "general_train" \
    --hem_extract_mode "hem-softmax_diff_small-offline" \
    --inference_fold "free" \
    --inference_interval 3000 \
    --experiments_sheet_dir "/OOB_RECOG/results-dev-test-robot-offline-general/mobilenet_general_rs" \
    --save_path "/OOB_RECOG/logs-dev-test-robot-offline-general/mobilenet_general_rs" > "./nohup_logs/mobilenet_general_rs-dev-test.out"