# - WS + proxy
# - WS + VI + proxy + all
# - WS + VI + proxy + neg
# - WS + VI + proxy + half + neg

nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-mean/hem-online-3-1-ws-proxy-update' > /dev/null


nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-proxy-update-repvgg' > /dev/null &




nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--multi_stage \
--n_stage 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-multi-stage-proxy-base-mobilenet' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--multi_stage \
--n_stage 3 \
--use_wise_sample \
--use_neg_proxy \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-multi-stage-neg-proxy-base-mobilenet' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--multi_stage \
--n_stage 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-multi-stage-proxy-base-resnet' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--multi_stage \
--n_stage 3 \
--use_wise_sample \
--use_neg_proxy \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-multi-stage-neg-proxy-base-resnet' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "multi-model" --max_epoch 100 \
--batch_size 128 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/multi-model-test' > /dev/null &





nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 7.7 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--multi_stage \
--n_stage 3 \
--experiment_type 'theator' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-theator2/theator-org' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--multi_stage \
--n_stage 3 \
--experiment_type 'theator' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-theator2/theator-our' > /dev/null &




nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 7.7 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--multi_stage \
--n_stage 3 \
--experiment_type 'theator' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-theator2/theator-org-ws' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--multi_stage \
--n_stage 3 \
--experiment_type 'theator' \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-theator2/theator-our-ws' > /dev/null &







nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--use_online_mcd \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs-unsup/proxy-all-norm' > /dev/null &