nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_proxy_all \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-all-proxy-update' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-proxy-update' > /dev/null &




nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_proxy_all \
--use_half_neg \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-all-half-proxy-update' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_half_neg \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-half-proxy-update' > /dev/null &




nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--use_online_mcd \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-ws-proxy-mcd-update' > /dev/null &



nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_online_mcd \
--sampling_type 1 --emb_type 2 --save_path '/code/OOB_Recog/logs/hem-online-3-1-proxy-mcd-update' > /dev/null &