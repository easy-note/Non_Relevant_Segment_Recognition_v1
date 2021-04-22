: << "END"
save results file name = {--title_name}-{--sub_title_name}.png

--model_name (yticks name)
--model_inference_path 
these args option input should be pair with ordering
if not, results are not synced.

--filter [median, mean]
--kernel_size [1,3,5,7,9,11,19], default = 1
only apply in predict results, not GT
if you dont want to apply filter, remove --filter and --kernel_size
END

: << "END"
R017
R017_ch1_video_01
R017_ch1_video_04

R022
R022_ch1_video_01
R022_ch1_video_03
R022_ch1_video_05

R116
R116_ch1_video_01
R116_ch1_video_03
R116_ch1_video_06

R208
R208_ch1_video_01
R208_ch1_video_03

R303
R303_ch1_video_01
R303_ch1_video_04
END


model_inf_path_1="./results-robot_oob_resnet18-1_1-last/R303/R303_ch1_video_04/Inference-R303_ch1_video_04.csv"
model_inf_path_2="./results-robot_oob_resnet34-1_1-last/R303/R303_ch1_video_04/Inference-R303_ch1_video_04.csv"
model_inf_path_3="./results-robot_oob_resnet50-1_1-last/R303/R303_ch1_video_04/Inference-R303_ch1_video_04.csv"
model_inf_path_4="./results-robot_oob_resnext50_32x4d-1_1-last/R303/R303_ch1_video_04/Inference-R303_ch1_video_04.csv"
model_inf_path_5="./results-robot_oob_wide_resnet50_2-1_1-last/R303/R303_ch1_video_04/Inference-R303_ch1_video_04.csv"

python frame_visualization.py \
--title_name "R303_Inference_RESNET_1_1" \
--sub_title_name "R303_ch1_video_04-MEDIAN(19)" \
--GT_path $model_inf_path_1 \
--model_name "resnet18" "resnet34" "resnet50" "resnext50_32x4d" "wide_resnet50_2" \
--model_infernce_path $model_inf_path_1 $model_inf_path_2 $model_inf_path_3 $model_inf_path_4 $model_inf_path_5 \
--results_save_dir "./visual_results/median" \
--filter "median" --kernel_size 19