for fold in 3; do
    for step in 5 10 15 20; do
        for factor in 0.9 0.5 0.1; do
            nohup python experiment_flow.py \
            --fold ${fold} \
            --model "mobilenet_v3_large" \
            --max_epoch 100 \
            --lr_scheduler "step_lr" \
            --lr_scheduler_step ${step} \
            --lr_scheduler_factor ${factor} \
            --cuda_list '3' \
            --save_path '/code/OOB_Recog/logs/schedule-step-mobile' \
            > /dev/null
        done
    done
done


for fold in 3; do
    for step in 5 10 15 20; do
        for factor in 0.9 0.5 0.1; do
            nohup python experiment_flow.py \
            --fold ${fold} \
            --model "efficientnet_b0" \
            --max_epoch 100 \
            --lr_scheduler "step_lr" \
            --lr_scheduler_step ${step} \
            --lr_scheduler_factor ${factor} \
            --cuda_list '3' \
            --save_path '/code/OOB_Recog/logs/schedule-step-efficient' \
            > /dev/null
        done
    done
done

for fold in 3; do
    for step in 5 10 15 20; do
        for factor in 0.9 0.5 0.1; do
            nohup python experiment_flow.py \
            --fold ${fold} \
            --model "resnet18" \
            --max_epoch 100 \
            --lr_scheduler "step_lr" \
            --lr_scheduler_step ${step} \
            --lr_scheduler_factor ${factor} \
            --cuda_list '3' \
            --save_path '/code/OOB_Recog/logs/schedule-step-resnet' \
            > /dev/null
        done
    done
done