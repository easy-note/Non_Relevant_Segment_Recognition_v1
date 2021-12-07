# Proxy - correct + WS
nohup python inference_only.py --fold '2' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '2' \
--WS_ratio 3 \
--use_wise_sample \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-proxy-trial:1-fold:2/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-proxy' > /dev/null &

nohup python inference_only.py --fold '3' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '3' \
--WS_ratio 3 \
--use_wise_sample \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-proxy-trial:1-fold:3/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-proxy' > /dev/null &


# Proxy - neg proxy + WS
nohup python inference_only.py --fold '2' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '2' \
--WS_ratio 3 \
--use_wise_sample \
--use_neg_proxy \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy-trial:1-fold:2/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy' > /dev/null &

nohup python inference_only.py --fold '3' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '3' \
--WS_ratio 3 \
--use_wise_sample \
--use_neg_proxy \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy-trial:1-fold:3/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy' > /dev/null &


# Proxy - neg proxy + WS + VI
nohup python inference_only.py --fold '2' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '2' \
--WS_ratio 3 \
--use_wise_sample \
--use_neg_proxy \
--use_online_mcd \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy-mcd-trial:1-fold:2/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy-mcd' > /dev/null &

nohup python inference_only.py --fold '3' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '3' \
--WS_ratio 3 \
--use_wise_sample \
--use_neg_proxy \
--use_online_mcd \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy-mcd-trial:1-fold:3/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-neg-proxy-mcd' > /dev/null &


# General + RS / WS
nohup python inference_only.py --fold '2' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '2' \
--WS_ratio 3 \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-general-trial:1-fold:2/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-general' > /dev/null &


nohup python inference_only.py --fold '3' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '3' \
--WS_ratio 3 \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-general-trial:1-fold:3/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-general' > /dev/null &


nohup python inference_only.py --fold '2' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '2' \
--WS_ratio 3 \
--use_wise_sample \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-general-trial:1-fold:2/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-general' > /dev/null &

nohup python inference_only.py --fold '3' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '3' \
--WS_ratio 3 \
--use_wise_sample \
--restore_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-general-trial:1-fold:3/TB_log/version_0' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-other-fold/resnet18-ws-general' > /dev/null &