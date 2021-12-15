# 21.11.26 off-line mc assets 뽑기
# [고정] hem_extract_mode:all-offline / random_seed:3829 / IB_ratio:3 / WS_ratio:2 / mc:5


nohup python ../visual_flow.py \
--fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/OOB_RECOG/logs/resnet18-ws-neg-proxy' \
--stage_flag \
--top_ratio 0.05 --n_dropout 5 \
--experiments_sheet_dir "/OOB_RECOG/results-onoff/resnet18-ws-neg-proxy" \
--restore_path '/OOB_RECOG/logs/resnet18-ws-neg-proxy-trial:1-fold:1/TB_log/version_0' > /dev/null