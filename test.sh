: << "END"
model_array=("resnet34" \
            "resnet50" \
            "wide_resnet50_2")

model_path_array=("./logs/robot/OOB/robot_oob_0406/ckpoint_robot_oob_0406-model=resnet34-batch=32-lr=0.001-epoch=49-last.ckpt" \
                "./logs/robot/OOB/robot_oob_0406/ckpoint_robot_oob_0406-model=resnet50-batch=32-lr=0.001-epoch=49-last.ckpt" \
                "./logs/robot/OOB/robot_oob_0406/ckpoint_robot_oob_0406-model=wide_resnet50_2-batch=32-lr=0.001-epoch=49-last.ckpt")

results_save_dir_array=("./results-robot_oob_resnet34-1_3-last" \
                "./results-robot_oob_resnet50-1_3-last" \
                "./results-robot_oob_wide_resnet50_2-1_3-last")


for (( i = 0 ; i < ${#model_path_array[@]} ; i++ ))
do
    python new_test_video.py \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/CAM_IO/robot/video" \
    --anno_dir "/data/CAM_IO/robot/OOB" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 10000 \
    --test_videos "R017" "R022" "R116" "R208" "R303"
done
END

model_array=("wide_resnet50_2" \
            "resnext50_32x4d" \
            "mobilenet_v3_small" \
            "squeezenet1_0")

model_path_array=("./logs/robot/OOB/robot-oob-0423-fold_2/ckpoint_robot-oob-0423-fold_2-model=wide_resnet50_2-batch=32-lr=0.001-fold=1-ratio=3-epoch=49-last.ckpt" \
                "./logs/robot/OOB/robot-oob-0423-fold_2/ckpoint_robot-oob-0423-fold_2-model=resnext50_32x4d-batch=32-lr=0.001-fold=1-ratio=3-epoch=49-last.ckpt" \
                "./logs/robot/OOB/robot-oob-0423-fold_2/ckpoint_robot-oob-0423-fold_2-model=mobilenet_v3_small-batch=32-lr=0.001-fold=1-ratio=3-epoch=49-last.ckpt" \
                "./logs/robot/OOB/robot-oob-0423-fold_2/ckpoint_robot-oob-0423-fold_2-model=squeezenet1_0-batch=32-lr=0.001-fold=3-ratio=1-epoch=49-last.ckpt")

results_save_dir_array=("./temp_results-robot_oob-wide_resnet50_2-fold_2-last" \
                "./temp_results-robot_oob-resnext50_32x4d-fold_2-last" \
                "./temp_results-robot_oob-mobilenet_v3_small-fold_2-last" \
                "./temp_results-robot_oob-squeezenet1_0-fold_2-last")

model_array=("wide_resnet50_2")
model_path_array=("./logs/robot/OOB/robot-oob-0423-fold_2/ckpoint_robot-oob-0423-fold_2-model=wide_resnet50_2-batch=32-lr=0.001-fold=2-ratio=3-epoch=49-last.ckpt")
results_save_dir_array=("./temp3_results-robot_oob-wide_resnet50_2-fold_2-last")


for (( i = 0 ; i < ${#model_path_array[@]} ; i++ ))
do
    python test.py \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/ROBOT/Video" \
    --anno_dir "/data/OOB" \
    --inference_assets_dir "/data/ROBOT/Inference" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 5 \
    --test_videos "R_17" "R_22" "R_116" "R_208" "R_303"
done