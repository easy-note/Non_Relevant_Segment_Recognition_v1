
model_array=("mobilenet_v3_small")

model_path_array=("/OOB_RECOG/logs/LAPA/OOB/model-scratch-mobilenet_v3_small-fold1/ckpoint_model-scratch-mobilenet_v3_small-fold1-model=mobilenet_v3_small-batch=32-lr=0.001-fold=1-ratio=3-epoch=26-last.ckpt")


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

model_path_array=("/OOB_RECOG/logs/ROBOT/OOB/V2-mobilenet_v3_large-all_new/ckpoint_0816-test-mobilenet_v3_large-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt")

results_save_dir_array=("./results_v2_new_lapa_oob-mobilenet_v3_large_from_robot-fold_1-last-fps1_60_all-new")


for (( i = 0 ; i < ${#model_path_array[@]} ; i++ ))
do
    python test.py \
    --mode "LAPA" \
    --assets_mode "DB" \
    --data_sheet_dir "./DATA_SHEET/NEW_LAPA" \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/LAPA/Video" \
    --anno_dir "/data/OOB" \
    --inference_assets_dir "/data/LAPA/Inference" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 60 \
    --test_videos 'L_325' 'L_412' 'L_443' 'L_450' 'L_507' 'L_522' 'L_535' 'L_550' 'L_605' 'L_661' 'L_669' 'L_676'
done

# 'R_2' 'R_6' 'R_13' 'R_74' 'R_100' 'R_202' 'R_301' 'R_302' 'R_311' 'R_312' 'R_313' 'R_336' 'R_362' 'R_363' 'R_386' 'R_405' 'R_418' 'R_423' 'R_424' 'R_526'
# 'L_605' 'L_305' 'L_477' 'L_430' 'L_340' 'L_475' 'L_393' 'L_569' 'L_484' 'L_419' 'L_427' 'L_556' 'L_408' 'L_539' 'L_433' 'L_423' 'L_572' 'L_654' 'L_385' 'L_545' 