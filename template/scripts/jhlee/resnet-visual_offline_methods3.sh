# 21.11.26 off-line mc assets 뽑기
# [고정] hem_extract_mode:all-offline / random_seed:3829 / IB_ratio:3 / WS_ratio:2 / mc:5


IB_ratio=3;

top_ratio=(0.10);
WS_ratio=2;
n_dropout=1;


for ratio in "${top_ratio[@]}";
do
    nohup python ../visual_flow.py \
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
        --hem_extract_mode "all-offline" \
        --experiments_sheet_dir "/OOB_RECOG/results/1203-generate-RESNET-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-experiment" \
        --save_path "/OOB_RECOG/logs/1203-generate-RESNET-IB_ratio=${IB_ratio}-ws_ratio=${WS_ratio}-experiment" > /dev/null
done;