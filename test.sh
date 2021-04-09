for model in "resnet18"
do
    python new_test_video.py \
    --model_path "/OOB_RECOG/logs/robot/OOB/robot_oob_0406/ckpoint_robot_oob_0406-model=resnet18-batch=32-lr=0.001-epoch=49-last.ckpt" \
    --data_dir "/data/CAM_IO/robot/video" \
    --anno_dir "/data/CAM_IO/robot/OOB" \
    --results_save_dir "./results_robot_oob_0406" \
    --model $model \
    --inference_step 1000 \
    --test_videos "R017" "R022" "R116" "R208" "R303"
done