:<<'END'
model_array=("mobilenet_v3_small" \
                   "squeezenet1_0" \
                   "mobilenet_v2")

project_name_array=("robot-oob-mobilenet_v3-1_3" \
                   "robot-oob-squeezenet1_0-1_3" \
                    "robot-oob-mobilenet_v2-1_3")


for (( i = 0 ; i < ${#model_array[@]} ; i++ ))
do
    python train_camIO.py \
    --project_name ${project_name_array[$i]} \
    --max_epoch 3 \
    --log_path "/OOB_RECOG/logs" \
    --batch_size 32 \
    --init_lr 1e-3 \
    --model ${model_array[$i]}
done
END


model_array=("wide_resnet50_2" \
            "resnext50_32x4d" \
            "mobilenet_v3_small" \
            "squeezenet1_0")

for (( i = 0 ; i < ${#model_array[@]} ; i++ ))
do
    python train_camIO.py \
    --project_name "robot-oob-0423-fold_1" \
    --max_epoch 50 \
    --log_path "/OOB_RECOG/logs" \
    --batch_size 32 \
    --init_lr 1e-3 \
    --model ${model_array[$i]} \
    --IB_ratio 3 \
    --random_seed 10 \
    --fold '1' \
    --num_gpus 1
done