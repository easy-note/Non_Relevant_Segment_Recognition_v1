for lr in 1e-2 1e-3 1e-4:
    do
        python train_camIO.py \
        --project_name="robot_oob_0402" \
        --max_epoch=50 \
        --log_path="/OOB_RECOG/logs" \
        --batch_size=32 \
        --init_lr=${lr}
    done
done

