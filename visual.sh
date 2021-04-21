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

model_inf_path_1="./results-robot_oob_resnet18-1_3-last/R017/R017_ch1_video_01/Inference_R017_ch1_video_01.csv"
model_inf_path_2="./results-robot_oob_resnet18-1_3-last/R017/R017_ch1_video_01/Inference_R017_ch1_video_01.csv"
model_inf_path_3="./results-robot_oob_resnet18-1_3-last/R017/R017_ch1_video_01/Inference_R017_ch1_video_01.csv"
model_inf_path_4="./results-robot_oob_resnet18-1_3-last/R017/R017_ch1_video_01/Inference_R017_ch1_video_01.csv"
model_inf_path_5="./results-robot_oob_resnet18-1_3-last/R017/R017_ch1_video_01/Inference_R017_ch1_video_01.csv"

python frame_visualization.py \
--title_name "R017_Inference_RESNET_1_3" \
--sub_title_name "R017_ch1_video_01" \
--GT_path $model_inf_path_1 \
--model_name "resnet18" "resnet34" "resnet50" "resnext50_32x4d" "wide_resnet50_2" \
--model_infernce_path $model_inf_path_1 $model_inf_path_2 $model_inf_path_3 $model_inf_path_4 $model_inf_path_5 \
--results_save_dir "./visual_results"