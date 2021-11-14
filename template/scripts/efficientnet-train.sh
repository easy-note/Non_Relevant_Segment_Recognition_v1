for trial in 1 2 3 4 5 6 7 8 9 10; do
    for fold in 1 2 3; do
        nohup python experiment_flow.py \
        --fold ${fold} \
        --trial ${trial} \
        --model "efficientnet_b0" \
        --max_epoch 150 \
        --batch_size 128 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list '1' \
        --generate_hem_mode 'normal' \
        --save_path '/code/OOB_Recog/logs/baseline-efficientnet' \
        > /dev/null
    done
done