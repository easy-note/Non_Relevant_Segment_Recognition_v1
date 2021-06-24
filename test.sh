
model_array=("mobilenet_v3_small")

model_path_array=("/OOB_RECOG/logs/LAPA/OOB/model-scratch-mobilenet_v3_small-fold1/ckpoint_model-scratch-mobilenet_v3_small-fold1-model=mobilenet_v3_small-batch=32-lr=0.001-fold=1-ratio=3-epoch=26-last.ckpt")

results_save_dir_array=("./results-last-lapa_oob-mobilenet_v3_small-fold_1-model-scratch_test")


for (( i = 0 ; i < ${#model_path_array[@]} ; i++ ))
do
    python test.py \
    --mode "LAPA" \
    --model_path ${model_path_array[$i]} \
    --data_dir "/data/LAPA/Video" \
    --anno_dir "/data/OOB" \
    --inference_assets_dir "/data/LAPA/Inference_test" \
    --results_save_dir ${results_save_dir_array[$i]} \
    --model ${model_array[$i]} \
    --inference_step 5 \
    --test_videos 'L_522'
done