for fold in 1 2 3; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "ig_resnext101_32x48d" \
    --max_epoch 150 \
    --batch_size 16 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --cuda_list '3' \
    --generate_hem_mode 'normal' \
    --save_path '/code/OOB_Recog/logs/baseline-resnext' \
    > /dev/null
done