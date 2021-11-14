for fold in 1 2 3; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "mobilenet_v3_large" \
    --cuda_list '0' \
    --save_path '/code/OOB_Recog/logs/baseline-mobile-torch' \
    --max_epoch 100 \
    --batch_size 128 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --train_method "normal" \
    > /dev/null
done


nohup python experiment_flow.py \
    --fold '1' \
    --model "mobilenet_v3_large" \
    --cuda_list '7' \
    --save_path '/code/OOB_Recog/logs/baseline-focus' \
    --max_epoch 200 \
    --batch_size 128 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --train_method "hem-focus" > /dev/null &


python experiment_flow.py \
    --fold '1' \
    --model "mobilenet_v3_large" \
    --cuda_list '7' \
    --save_path '/code/OOB_Recog/logs/baseline-focus2' \
    --max_epoch 200 \
    --batch_size 128 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --train_method "hem-focus2"


nohup python experiment_flow.py \
    --fold '1' \
    --model "mobilenet_v3_large" \
    --cuda_list '4' \
    --save_path '/code/OOB_Recog/logs/baseline-mobile-torch-test-emb4' \
    --max_epoch 200 \
    --batch_size 128 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --train_method "hem-emb4" \
    > /dev/null &

nohup python experiment_flow.py \
    --fold '1' \
    --model "mobilenetv3_large_100" \
    --cuda_list '4' \
    --save_path '/code/OOB_Recog/logs/baseline-mobile-timm2-test-emb4' \
    --max_epoch 200 \
    --batch_size 128 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --train_method "hem-emb4" \
    > /dev/null &

# for fold in 1 2 3; do
#     nohup python experiment_flow.py \
#     --fold ${fold} \
#     --model "mobilenet_v3_large" \
#     --cuda_list '1' \
#     --save_path '/code/OOB_Recog/logs/baseline-mobile-torch-emb-ver3' \
#     --max_epoch 100 \
#     --batch_size 128 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --train_method "hem-emb3" \
#     > /dev/null
# done


# python experiment_flow.py \
#     --fold '1' \
#     --model "mobilenet_v3_large" \
#     --cuda_list '1' \
#     --save_path '/code/OOB_Recog/logs/baseline-mobile-torch-emb-ver3' \
#     --max_epoch 100 \
#     --batch_size 128 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --train_method "hem-emb3"

# for fold in 1 2 3; do
#     nohup python experiment_flow.py \
#     --fold ${fold} \
#     --model "mobilenetv3_large_100_miil" \
#     --cuda_list '1' \
#     --save_path '/code/OOB_Recog/logs/baseline-mobile-timm' \
#     --max_epoch 100 \
#     --batch_size 128 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --train_method "normal" \
#     > /dev/null
# done

# for fold in 1 2 3; do
#     nohup python experiment_flow.py \
#     --fold ${fold} \
#     --model "mobilenetv3_large_100_miil" \
#     --cuda_list '3' \
#     --save_path '/code/OOB_Recog/logs/baseline-mobile-timm-emb-ver2' \
#     --max_epoch 100 \
#     --batch_size 128 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --train_method "hem-emb2" \
#     > /dev/null
# done

for fold in 1 2 3; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "mobilenetv3_large_100" \
    --cuda_list '5' \
    --save_path '/code/OOB_Recog/logs/baseline-mobile-timm2-emb-ver2' \
    --max_epoch 100 \
    --batch_size 128 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --train_method "hem-emb2" \
    > /dev/null
done

# nohup python experiment_flow.py \
#     --fold '1' \
#     --model "mobilenet_v3_large" \
#     --cuda_list '0' \
#     --save_path '/code/OOB_Recog/logs/baseline-mobile-torch-test' \
#     --max_epoch 200 \
#     --batch_size 128 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --train_method "normal" > /dev/null &


# for fold in 1 2 3; do
#     nohup python experiment_flow.py \
#     --fold ${fold} \
#     --model "efficientnet_b0" \
#     --cuda_list '1' \
#     --save_path '/code/OOB_Recog/logs/baseline' \
#     --max_epoch 50 \
#     --batch_size 128 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     > /dev/null
# done

# for fold in 1 2 3; do
#     nohup python experiment_flow.py \
#     --fold ${fold} \
#     --model "resnet18" \
#     --cuda_list '1' \
#     --save_path '/code/OOB_Recog/logs/baseline' \
#     --max_epoch 50 \
#     --batch_size 128 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     > /dev/null
# done