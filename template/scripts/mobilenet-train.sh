for trial in 1 2 3 4 5 6 7 8 9 10; do
    for fold in 1 2 3; do
        nohup python inference_flow.py \
        --fold ${fold} \
        --trial ${trial} \
        --model "mobilenetv3_large_100" \
        --max_epoch 150 \
        --batch_size 128 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '2' \
        --train_mode 'normal' \
        --generate_hem_mode 'normal' \
        --save_path '/code/OOB_Recog/logs/baseline-mobilenet' \
        > /dev/null
    done
done




python inference_flow.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 1 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '2' \
        --stage 'general_train' \
        --save_path '/code/OOB_Recog/logs/baseline-mobilenet'

python inference_flow.py \
        --fold '1' \
        --trial 1 \
        --model "mobilenetv3_large_100" \
        --max_epoch 1 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '2' \
        --stage 'hem_train' \
        --hem_extract_mode 'hem-emb3-online' \
        --save_path '/code/OOB_Recog/logs/baseline-mobilenet'