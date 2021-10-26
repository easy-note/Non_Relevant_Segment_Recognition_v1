# for fold in 3; do
#     for step in 5 10 15 20; do
#         for factor in 0.9 0.5 0.1; do
#             nohup python experiment_flow.py \
#             --fold ${fold} \
#             --model "mobilenet_v3_large" \
#             --max_epoch 100 \
#             --lr_scheduler "reduced" \
#             --lr_scheduler_step ${step} \
#             --lr_scheduler_factor ${factor} \
#             --cuda_list '0' \
#             --save_path '/code/OOB_Recog/logs/schedule-reduced-mobile' \
#             > /dev/null
#         done
#     done
# done


# for fold in 3; do
#     for step in 5 10 15 20; do
#         for factor in 0.9 0.5 0.1; do
#             nohup python experiment_flow.py \
#             --fold ${fold} \
#             --model "efficientnet_b0" \
#             --max_epoch 100 \
#             --lr_scheduler "reduced" \
#             --lr_scheduler_step ${step} \
#             --lr_scheduler_factor ${factor} \
#             --cuda_list '1' \
#             --save_path '/code/OOB_Recog/logs/schedule-reduced-efficient' \
#             > /dev/null
#         done
#     done
# done

# for fold in 3; do
#     for step in 5 10 15 20; do
#         for factor in 0.9 0.5 0.1; do
#             nohup python experiment_flow.py \
#             --fold ${fold} \
#             --model "resnet18" \
#             --max_epoch 100 \
#             --lr_scheduler "reduced" \
#             --lr_scheduler_step ${step} \
#             --lr_scheduler_factor ${factor} \
#             --cuda_list '2' \
#             --save_path '/code/OOB_Recog/logs/schedule-reduced-resnet' \
#             > /dev/null
#         done
#     done
# done


for fold in 3; do
    for step in 5 10 15 20; do
        for factor in 0.9 0.5 0.1; do
            nohup python experiment_flow.py \
            --fold ${fold} \
            --model "swin_large_patch4_window7_224" \
            --max_epoch 100 \
            --batch_size 64 \
            --lr_scheduler "reduced" \
            --lr_scheduler_step ${step} \
            --lr_scheduler_factor ${factor} \
            --cuda_list '7' \
            --save_path '/code/OOB_Recog/logs/schedule-reduced-swin' \
            > /dev/null
        done
    done
done