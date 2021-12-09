# 21.11.26 off-line mc assets 뽑기
# [고정] hem_extract_mode:all-offline / random_seed:3829 / IB_ratio:3 / WS_ratio:2 / mc:5


IB_ratio=3;

top_ratio=(0.10);
WS_ratio=3;
n_dropout=5;


for ratio in "${top_ratio[@]}";
do
    python ../visual_flow.py \
        --stage_flag \
        --fold "1" \
        --trial 1 \
        --use_wise_sample \
        --WS_ratio ${WS_ratio} \
        --model "resnet18" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 50 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "5" \
        --random_seed 3829 \
        --IB_ratio ${IB_ratio} \
        --inference_fold "1" \
        --top_ratio ${ratio} \
        --n_dropout ${n_dropout} \
        --stage "hem_train" \
        --hem_per_patient \
        --hem_extract_mode "hem-softmax-offline" \
        --experiments_sheet_dir "/OOB_RECOG/results/STAGE_TEST-generate-RESNET-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-experiment" \
        --save_path "/OOB_RECOG/logs/STAGE_TEST-generate-RESNET-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-experiment"
done;