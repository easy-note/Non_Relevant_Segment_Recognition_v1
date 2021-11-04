for trial in 1 2 3 4 5 6 7 8 9 10; do
    for fold in 1 2 3; do
        nohup python inference_flow.py \
        --fold ${fold} \
        --trial ${trial} \
        --experiment_type "theator" \
        --model "resnet18" \
        --IB_ratio 3 \
        --optimizer "sgd" \
        --batch_size 64 \
        --lr_scheduler "mul_step_lr" \
        --max_epoch 120 \
        --cuda_list '4' \
        --save_path '/code/OOB_Recog/logs/theator-our-ratio' \
        > /dev/null
    done


    for fold in 1 2 3; do
        nohup python inference_flow.py \
        --fold ${fold} \
        --trial ${trial} \
        --experiment_type "theator" \
        --model "resnet18" \
        --IB_ratio 7.7 \
        --optimizer "sgd" \
        --batch_size 64 \
        --lr_scheduler "mul_step_lr" \
        --max_epoch 120 \
        --cuda_list '4' \
        --save_path '/code/OOB_Recog/logs/theator-org-ratio' \
        > /dev/null
    done
done