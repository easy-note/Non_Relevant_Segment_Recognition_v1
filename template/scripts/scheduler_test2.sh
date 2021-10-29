# for fold in 3; do
#     for max_iter in 1000 2000 5000 10000 20000 50000; do
#         nohup python experiment_flow.py \
#         --fold ${fold} \
#         --model "mobilenet_v3_large" \
#         --max_epoch 100 \
#         --lr_scheduler "cosine" \
#         --t_max_iter ${max_iter} \
#         --cuda_list '3' \
#         --save_path '/code/OOB_Recog/logs/schedule-cosine-mobile' \
#         > /dev/null
#     done
# done


# for fold in 3; do
    # for max_iter in 1000 2000 5000 10000 20000 50000; do
    #     nohup python experiment_flow.py \
    #     --fold ${fold} \
    #     --model "efficientnet_b0" \
    #     --max_epoch 100 \
    #     --lr_scheduler "cosine" \
    #     --t_max_iter ${max_iter} \
    #     --cuda_list '4' \
    #     --save_path '/code/OOB_Recog/logs/schedule-cosine-efficient' \
    #     > /dev/null
    # done
# done

# for fold in 3; do
#     for max_iter in 1000 2000 5000 10000 20000 50000; do
#         nohup python experiment_flow.py \
#         --fold ${fold} \
#         --model "resnet18" \
#         --max_epoch 100 \
#         --lr_scheduler "cosine" \
#         --t_max_iter ${max_iter} \
#         --cuda_list '5' \
#         --save_path '/code/OOB_Recog/logs/schedule-cosine-resnet' \
#         > /dev/null
#     done
# done


for fold in 3; do
    for max_iter in 1000 2000 5000 10000 20000 50000; do
        nohup python experiment_flow.py \
        --fold ${fold} \
        --model "swin_large_patch4_window7_224" \
        --max_epoch 100 \
        --batch_size 64 \
        --lr_scheduler "cosine" \
        --t_max_iter ${max_iter} \
        --cuda_list '4' \
        --save_path '/code/OOB_Recog/logs/schedule-cosine-swin' \
        > /dev/null
    done
done