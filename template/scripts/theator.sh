for fold in 1 2 3 4 5; do
    nohup python experiment_flow_theator.py \
    --fold ${fold} \
    --experiment_type "theator" \
    --IB_raio 3 \
    --save_path '../logs/theator-our-ratio' \
    > /dev/null
done


for fold in 1 2 3 4 5; do
    nohup python experiment_flow_theator.py \
    --fold ${fold} \
    --experiment_type "theator" \
    --IB_raio 7.7 \
    --save_path '../logs/theator-org-ratio' \
    > /dev/null
done