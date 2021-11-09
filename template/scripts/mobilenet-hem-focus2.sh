for trial in 1; do
    for fold in 1 2 3; do
        nohup python inference_flow.py \
        --fold ${fold} \
        --trial ${trial} \
        --model "mobilenetv3_large_100" \
        --max_epoch 150 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '3' \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-focus2-online' \
        --save_path '/code/OOB_Recog/logs/hem-focus-mobilenet-ver2' \
        > /dev/null
    done
done


nohup python visual_flow.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '3' \
        --random_seed 3829 \
        --IB_ratio 3 \
        --loss_fn 'focal' \
        --stage 'general_train' \
        --save_path '/code/OOB_Recog/logs/general-3-1-focal' > /dev/null &


nohup python visual_flow.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '4' \
        --IB_ratio 3 \
        --random_seed 3829 \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-focus-online' \
        --sampling_type 4 \
        --inference_fold '1' \
        --save_path '/code/OOB_Recog/logs/hem-online-ws4-3-1-step' > /dev/null &


nohup python visual_flow.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 6 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '6' \
        --IB_ratio 3 \
        --random_seed 3829 \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-focus-online' \
        --sampling_type 1 \
        --use_meta \
        --inference_fold '1' \
        --save_path '/code/OOB_Recog/logs/hem-online-ws1-3-1-step-meta2' > /dev/null &

nohup python visual_flow.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '5' \
        --IB_ratio 5 \
        --random_seed 10 \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-emb-online' \
        --inference_fold '1' \
        --save_path '/code/OOB_Recog/logs/hem-online-5-1-step-org-seed' > /dev/null &


nohup python inference_only.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "cosine" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '3' \
        --IB_ratio 3 \
        --random_seed 3829 \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-emb-online' \
        --inference_fold '1' \
        --save_path '/code/OOB_Recog/logs/hem-online-3-1-cosine' \
        --restore_path '/code/OOB_Recog/logs/hem-online-3-1-cosine-trial:1-fold:1/TB_log/version_0' > /dev/null &


nohup python inference_only.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "reduced" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '2' \
        --IB_ratio 3 \
        --random_seed 3829 \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-focus-online' \
        --sampling_type 3 \
        --inference_fold '1' \
        --save_path '/code/OOB_Recog/logs/hem-online-5-1-step-org-seed' \
        --restore_path '/code/OOB_Recog/logs/hem-online-5-1-step-org-seed-trial:1-fold:1/TB_log/version_0' \
        > /dev/null &


nohup python inference_only.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '3' \
        --IB_ratio 5 \
        --random_seed 3829 \
        --stage 'general_train' \
        --sampling_type 3 \
        --inference_fold '1' \
        --save_path '/code/OOB_Recog/logs/general-3-1-focal' \
        --restore_path '/code/OOB_Recog/logs/general-3-1-focal-trial:1-fold:1/TB_log/version_0' \
        > /dev/null &

