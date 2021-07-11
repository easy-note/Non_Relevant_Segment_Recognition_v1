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
            "squeezenet1_0" \
            "mobilenet_v3_large")

model_path_array=("./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_wide_resnet50_2-fold3-model=wide_resnet50_2-batch=32-lr=0.001-fold=3-ratio=3-epoch=24-last.ckpt" \
                "./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_resnext50_32x4d-fold3-model=resnext50_32x4d-batch=32-lr=0.001-fold=3-ratio=3-epoch=32-last.ckpt" \
                "./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_mobilenet_v3_small-fold3-model=mobilenet_v3_small-batch=32-lr=0.001-fold=3-ratio=3-epoch=24-last.ckpt" \
                "./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_squeezenet1_0-fold3-model=squeezenet1_0-batch=32-lr=0.001-fold=3-ratio=3-epoch=24-last.ckpt" \
                "/OOB_RECOG/logs/ROBOT/OOB/mobilenet_v3_large/ckpoint_mobilenet_v3_large-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt")

results_save_dir_array=("./temp_results-robot_oob-wide_resnet50_2-fold_3-last" \
                "./temp_results-robot_oob-resnext50_32x4d-fold_3-last" \
                "./temp_results-robot_oob-mobilenet_v3_small-fold_3-last" \
                "./temp_results-robot_oob-squeezenet1_0-fold_3-last" \
                "./results_v2_robot_oob-mobilenet_v3_large-fold_1-last")


for (( i = 4 ; i < ${#model_path_array[@]} ; i++ ))
do
    python test.py \
    --mode "ROBOT" \
    --assets_mode "DB" \
    --data_sheet_dir "./DATA_SHEET" \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/LAPA/Video" \
    --anno_dir "/data/OOB" \
    --inference_assets_dir "/data/LAPA/Inference" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 5 \
    --test_videos 'R_2' 'R_6' 'R_13' 'R_74' 'R_100' 'R_202' 'R_301' 'R_302' 'R_311' 'R_312' 'R_313' 'R_336' 'R_362' 'R_363' 'R_386' 'R_405' 'R_418' 'R_423' 'R_424' 'R_526'
done
