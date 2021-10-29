for fold in 1 2 3; do
    nohup python experiment_flow_theator.py \
    --fold ${fold} \
    --experiment_type "theator" \
    --IB_raio 3 \
    --cuda_list '3' \
    --save_path '/code/OOB_Recog/logs/theator-our-ratio' \
    > /dev/null
done


for fold in 1 2 3; do
    nohup python experiment_flow_theator.py \
    --fold ${fold} \
    --experiment_type "theator" \
    --IB_raio 7.7 \
    --cuda_list '3' \
    --save_path '/code/OOB_Recog/logs/theator-org-ratio' \
    > /dev/null
done