# for fold in 1 2 3 4 5; do
#     nohup python experiment_flow.py \
#     --fold ${fold} \
#     --model "mobilenet_v3_large" \
#     > /dev/null
# done

for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "efficientnet_b0" \
    > /dev/null
done

for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "resnet18" \
    > /dev/null
done