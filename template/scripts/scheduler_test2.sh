# for fold in 3; do
#     for step in 5 10 15 20; do
#         for factor in 0.9 0.5 0.1; do
#             for max_iter in 1000 2000 5000 10000 20000 50000; do
#                 nohup python experiment_flow.py \
#                 --fold ${fold} \
#                 --model "mobilenet_v3_large" \
#                 --max_epoch 100 \
#                 --lr_scheduler "cosine" \
#                 --lr_scheduler_step ${step} \
#                 --lr_scheduler_factor ${factor} \
#                 --t_max_iter ${max_iter} \
#                 --cuda_list '3' \
#                 --save_path '/code/OOB_Recog/logs/schedule-cosine-mobile' \
#                 > /dev/null
#             done
#         done
#     done
# done


# for fold in 3; do
#     for step in 5 10 15 20; do
#         for factor in 0.9 0.5 0.1; do
#             for max_iter in 1000 2000 5000 10000 20000 50000; do
#                 nohup python experiment_flow.py \
#                 --fold ${fold} \
#                 --model "efficientnet_b0" \
#                 --max_epoch 100 \
#                 --lr_scheduler "cosine" \
#                 --lr_scheduler_step ${step} \
#                 --lr_scheduler_factor ${factor} \
#                 --t_max_iter ${max_iter} \
#                 --cuda_list '4' \
#                 --save_path '/code/OOB_Recog/logs/schedule-cosine-efficient' \
#                 > /dev/null
#             done
#         done
#     done
# done

for fold in 3; do
    for step in 5 10 15 20; do
        for factor in 0.9 0.5 0.1; do
            for max_iter in 1000 2000 5000 10000 20000 50000; do
                nohup python experiment_flow.py \
                --fold ${fold} \
                --model "resnet18" \
                --max_epoch 100 \
                --lr_scheduler "cosine" \
                --lr_scheduler_step ${step} \
                --lr_scheduler_factor ${factor} \
                --t_max_iter ${max_iter} \
                --cuda_list '5' \
                --save_path '/code/OOB_Recog/logs/schedule-cosine-resnet' \
                > /dev/null
            done
        done
    done
done