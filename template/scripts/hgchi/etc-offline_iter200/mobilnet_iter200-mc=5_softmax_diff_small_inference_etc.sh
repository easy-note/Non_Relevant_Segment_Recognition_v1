# 22-02-23 Inference only etc

# 해당 resotre path 에 model 경로 넣어주세요
# 해당 경로 내 inference_etc_results 라는 이름으로 Inference 결과가 기록됩니다.
restore_path="/OOB_RECOG/logs_robot-offline-iter200/mobilenet_iter200_apply-MC=5-softmax_diff_small/TB_log/version_4"

# 이건 experiments_sheet 에 기록할때 구분하고자 그냥 넣었습니다. (+ random_seed, stage, methods)
IB_ratio=3;
top_ratio=(0.05);
WS_ratio=3;

# etc의 경우 다음을 꼭 지켜주세요
# inference_fold "free"
# hem_extract_mode 의 경우 checkpoint model 불러올때, offline, online 을 구분하므로 작성해주세요
# 이외 model 불러올때 사용하시는 args 도 함께 전달해주세요.

for ratio in "${top_ratio[@]}";
do
    nohup python -u ../../inference_only_etc.py \
        --restore_path ${restore_path} \
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
        --cuda_list "4" \
        --random_seed 3829 \
        --IB_ratio ${IB_ratio} \
        --train_stage "hem_train" \
        --hem_extract_mode "hem-softmax_diff_small-offline" \
        --inference_fold "free" \
        --inference_interval 30 \
        --experiments_sheet_dir "/OOB_RECOG/results_robot-offline-iter200/mobilenet_iter200_inference_etc-MC=5" > "./nohup_logs/mobilenet_iter200_inference_etc-MC=5-softmax_diff_small.out"
done;