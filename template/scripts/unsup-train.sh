################# unsupervised dataset
# general soft label trainset -> train 




nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '7' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' \
--inference_fold '1' \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--update_type 4 \
--update_type2 True \
--use_step_weight True \
--experiment_sub_type 'unsup-general' \
--restore_path '/code/OOB_Recog/logs-ws-proxy-type2-4-trial:1-fold:1/TB_log/version_0'
--save_path '/code/OOB_Recog/logs-unsup/unsup-general-ws-train-proxy' > /dev/null &










nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' \
--hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-general' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-general-ws-train-proxy-base' > /dev/null &

nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' \
--hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--use_neg_proxy \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-general' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-general-ws-train-proxy-wrong' > /dev/null &


# proxy-base soft label trainset -> train 
nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-proxy' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-proxy-base-ws-train-general' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' \
--hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-proxy' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-proxy-base-ws-train-proxy-base' > /dev/null &

nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '5' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' \
--hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--use_neg_proxy \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-proxy' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-proxy-base-ws-train-proxy-wrong' > /dev/null &


# proxy-wrong soft label trainset -> train 
nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '5' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-proxy-wrong' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-proxy-wrong-ws-train-general' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '6' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' \
--hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-proxy-wrong' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-proxy-wrong-ws-train-proxy-base' > /dev/null &

nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" \
--max_epoch 100 --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 \
--lr_scheduler_factor 0.9 --cuda_list '7' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' \
--hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--use_neg_proxy \
--WS_ratio 3 --use_wise_sample \
--multi_stage --n_stage 3 \
--sampling_type 1 --emb_type 2 \
--experiment_sub_type 'unsup-proxy-wrong' \
--save_path '/code/OOB_Recog/logs-unsup/unsup-proxy-wrong-ws-train-proxy-wrong' > /dev/null &