for model in "mobilenet_v2"
do
    python train_camIO.py \
    --project_name "robot_oob_0408" \
    --max_epoch 50 \
    --log_path "/OOB_RECOG/logs" \
    --batch_size 32 \
    --init_lr 1e-3 \
    --model $model
done