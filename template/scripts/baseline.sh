for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "mobilenet_v3_large" \
    --cuda_list '1' \
    --save_path '../logs/baseline' \
    > /dev/null
done

for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "efficientnet_b0" \
    --cuda_list '1' \
    --save_path '../logs/baseline' \
    > /dev/null
done

for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "resnet18" \
    --cuda_list '1' \
    --save_path '../logs/baseline' \
    > /dev/null
done