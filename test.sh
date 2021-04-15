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

model_array=("resnet34" \
            "resnet50" \
            "wide_resnet50_2")

model_path_array=("./logs/robot/OOB/robot_oob_0406/ckpoint_robot_oob_0406-model=resnet34-batch=32-lr=0.001-epoch=49-last.ckpt" \
                "./logs/robot/OOB/robot_oob_0406/ckpoint_robot_oob_0406-model=resnet50-batch=32-lr=0.001-epoch=49-last.ckpt" \
                "./logs/robot/OOB/robot_oob_0406/ckpoint_robot_oob_0406-model=wide_resnet50_2-batch=32-lr=0.001-epoch=49-last.ckpt")

results_save_dir_array=("./new_results-robot_oob_resnet34-1_3-last" \
                "./new_results-robot_oob_resnet50-1_3-last" \
                "./new_results-robot_oob_wide_resnet50_2-1_3-last")


for (( i = 1 ; i < ${#model_path_array[@]} ; i++ ))
do
    python renewal_test_video.py \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/CAM_IO/robot/video" \
    --anno_dir "/data/CAM_IO/robot/OOB" \
    --inference_assets_dir "/data/CAM_IO/robot/inference_dataset" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 5 \
    --test_videos "R017" "R022" "R116" "R208" "R303"
done