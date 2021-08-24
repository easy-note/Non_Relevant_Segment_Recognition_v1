: << "END"
parser.add_argument('--title_name', type=str, help='plot title')

parser.add_argument('--model_path', type=str, help='model ckpt path')

parser.add_argument('--model_name', type=str, nargs='+',
					choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d', 'mobilenet_v2', 'mobilenet_v3_small', 'squeezenet1_0'], help='trained backborn model, it will be yticks name')

parser.add_argument('--inference_img_dir', type=str, help='root dir for inference img')

parser.add_argument('--save_dir', type=str, help='gradcam results save path')
END

model_name="wide_resnet50_2"

python visual_gradcam.py \
--title_name "${model_name}_FP" \
--model_path "./logs/ROBOT/OOB/robot-oob-0423-fold_2/ckpoint_robot-oob-0423-fold_2-model=wide_resnet50_2-batch=32-lr=0.001-fold=2-ratio=3-epoch=49-last.ckpt" \
--model_name ${model_name} \
--inference_img_dir "./results-robot_oob-wide_resnet50_2-fold_2-last/R006/R006_ch1_video_01/fp_frame" \
--save_dir "./gradcam_results"


