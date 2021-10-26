for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "mobilenet_v3_large" \
    --train_method "hem-emb" \
    --save_path '../logs/emb' \
    > /dev/null
done

for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "efficientnet_b0" \
    --train_method "hem-emb" \
    --save_path '../logs/emb' \
    > /dev/null
done

for fold in 1 2 3 4 5; do
    nohup python experiment_flow.py \
    --fold ${fold} \
    --model "resnet18" \
    --train_method "hem-emb" \
    --save_path '../logs/emb' \
    > /dev/null
done
