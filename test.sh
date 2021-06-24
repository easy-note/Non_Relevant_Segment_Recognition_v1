
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

model_array=("wide_resnet50_2" \
            "resnext50_32x4d" \
            "mobilenet_v3_small" \
            "squeezenet1_0")

model_path_array=("./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_wide_resnet50_2-fold3-model=wide_resnet50_2-batch=32-lr=0.001-fold=3-ratio=3-epoch=24-last.ckpt" \
                "./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_resnext50_32x4d-fold3-model=resnext50_32x4d-batch=32-lr=0.001-fold=3-ratio=3-epoch=32-last.ckpt" \
                "./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_mobilenet_v3_small-fold3-model=mobilenet_v3_small-batch=32-lr=0.001-fold=3-ratio=3-epoch=24-last.ckpt" \
                "./logs/LAPA/OOB/lapa-oob-0608-fold_3/ckpoint_squeezenet1_0-fold3-model=squeezenet1_0-batch=32-lr=0.001-fold=3-ratio=3-epoch=24-last.ckpt")

results_save_dir_array=("./temp_results-robot_oob-wide_resnet50_2-fold_3-last" \
                "./temp_results-robot_oob-resnext50_32x4d-fold_3-last" \
                "./temp_results-robot_oob-mobilenet_v3_small-fold_3-last" \
                "./temp_results-robot_oob-squeezenet1_0-fold_3-last")


for (( i = 0 ; i < ${#model_path_array[@]} ; i++ ))
do
    python test.py \
    --mode "LAPA" \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/LAPA/Video" \
    --anno_dir "/data/OOB" \
    --inference_assets_dir "/data/LAPA/Inference" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 5 \
    --test_videos 'L_450' 'L_669' 'L_676' 'L_535' 'L_496'
done