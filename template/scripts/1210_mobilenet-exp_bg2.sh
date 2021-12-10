
# Proxy - neg proxy + WS
nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--use_neg_proxy \
--sampling_type 1 --emb_type 2 --save_path '/OOB_RECOG/logs/mobilenet-ws-neg-proxy' \
--stage_flag \
--top_ratio 0.05 --n_dropout 5 > /dev/null

# # Proxy - neg proxy + WS
# python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 2 \
# --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
# --IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
# --WS_ratio 3 \
# --use_wise_sample \
# --use_neg_proxy \
# --sampling_type 1 --emb_type 2 --save_path '/OOB_RECOG/logs/mobilenet-ws-neg-proxy' \
# --use_test_batch \
# --stage_flag \
# --top_ratio 0.05 --n_dropout 5