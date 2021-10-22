for fold in 1 2 3 4 5; do
    nohup python experiment_flow_theator.py \
    --fold ${fold} \
    > /dev/null
done