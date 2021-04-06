for model in "resnet18" "resnet34" "resnet50"
do
    python train_camIO.py \
    --project_name="robot_oob_0406" \
    --max_epoch=50 \
    --log_path="/OOB_RECOG/logs" \
    --batch_size=32 \
    --init_lr=1e-3 \
    --model=$model
done