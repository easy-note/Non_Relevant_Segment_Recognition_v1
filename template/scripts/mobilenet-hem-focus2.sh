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
        --cuda_list '2' \
        --random_seed 3829 \
        --IB_ratio 5 \
        --stage 'general_train' \
        --save_path '/code/OOB_Recog/logs/general-5-1-no-focal' > /dev/null &


nohup python visual_flow.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '7' \
        --IB_ratio 3 \
        --random_seed 3829 \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-focus-online' \
        --sampling_type 3 \
        --save_path '/code/OOB_Recog/logs/hem-online-ws3-3-1-step' > /dev/null &

