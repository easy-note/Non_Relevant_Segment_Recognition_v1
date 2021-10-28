for fold in 1 2 3; do
    for step in 5 10; do
        for factor in 0.9 0.5 0.1; do
            nohup python experiment_flow.py \
            --fold ${fold} \
            --model "swin_large_patch4_window7_224" \
            --max_epoch 200 \
            --batch_size 64 \
            --lr_scheduler "reduced" \
            --lr_scheduler_step ${step} \
            --lr_scheduler_factor ${factor} \
            --cuda_list '5' \
            --save_path '/code/OOB_Recog/logs/schedule-reduced-swin' \
            > /dev/null
        done
    done
done