
# # Proxy - correct + WS
nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '6' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--sampling_type 1 --emb_type 2 --save_path '/OOB_RECOG/logs/resnet18-ws-proxy' \
--stage_flag \
--top_ratio 0.10 --n_dropout 5 > /dev/null


# # Proxy - correct + WS
# python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 2 \
# --batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '6' \
# --IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
# --WS_ratio 3 \
# --use_wise_sample \
# --sampling_type 1 --emb_type 2 --save_path '/OOB_RECOG/logs/resnet18-ws-proxy' \
# --use_test_batch \
# --stage_flag \
# --top_ratio 0.10 --n_dropout 5
