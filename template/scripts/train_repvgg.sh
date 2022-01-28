# - WS + proxy
# - WS + VI + proxy + all
# - WS + VI + proxy + neg
# - WS + VI + proxy + half + neg

nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_neg_proxy \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-neg-proxy-update-repvgg-a0' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--use_neg_proxy \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-neg-proxy-update-repvgg-a0' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_neg_proxy \
--use_wise_sample \
--use_online_mcd \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-neg-proxy-mcd-update-repvgg-a0' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--use_proxy_all \
--use_online_mcd \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-all-proxy-mcd-update-repvgg-a0' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-general-repvgg-a0' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-general-repvgg-a0' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "repvgg-a0" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-proxy-update-repvgg-a0' > /dev/null &