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

model_array=("mobilenet_v3_large")

model_path_array=("/OOB_RECOG/logs/ROBOT/OOB/V1-mobilenet_v3_large/ckpoint_V1-mobilenet_v3_large-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt")

results_save_dir_array=("./results_v1_new_lapa_oob-mobilenet_v3_large_from_robot-fold_1-last")


for (( i = 0 ; i < ${#model_path_array[@]} ; i++ ))
do
    python test.py \
    --mode "LAPA" \
    --assets_mode "DB" \
    --data_sheet_dir "./DATA_SHEET" \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/LAPA/Video" \
    --anno_dir "/data/OOB" \
    --inference_assets_dir "/data/LAPA/Inference" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 5 \
    --test_videos 'L_605' 'L_305' 'L_477' 'L_430' 'L_340' 'L_475' 'L_393' 'L_569' 'L_484' 'L_419' 'L_427' 'L_556' 'L_408' 'L_539' 'L_433' 'L_423' 'L_572' 'L_654' 'L_385' 'L_545' 
done

# 'R_2' 'R_6' 'R_13' 'R_74' 'R_100' 'R_202' 'R_301' 'R_302' 'R_311' 'R_312' 'R_313' 'R_336' 'R_362' 'R_363' 'R_386' 'R_405' 'R_418' 'R_423' 'R_424' 'R_526'
# 'L_605' 'L_305' 'L_477' 'L_430' 'L_340' 'L_475' 'L_393' 'L_569' 'L_484' 'L_419' 'L_427' 'L_556' 'L_408' 'L_539' 'L_433' 'L_423' 'L_572' 'L_654' 'L_385' 'L_545' 