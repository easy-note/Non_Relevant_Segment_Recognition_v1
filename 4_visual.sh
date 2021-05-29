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
R17
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


patient_array=("R_17" \
            "R_22" \
            "R_116" \
            "R_208" \
            "R_303")

patient_array=("R_17")

for (( i = 0 ; i < ${#patient_array[@]} ; i++ ))
do
    model_inf_path_1="./temp2_results-robot_oob-wide_resnet50_2-fold_2-last/${patient_array[$i]}/Inference-${patient_array[$i]}.csv"
    model_inf_path_2="./temp2_results-robot_oob-wide_resnet50_2-fold_2-last/${patient_array[$i]}/Inference-${patient_array[$i]}.csv"
    model_inf_path_3="./temp2_results-robot_oob-wide_resnet50_2-fold_2-last/${patient_array[$i]}/Inference-${patient_array[$i]}.csv"
    model_inf_path_4="./temp2_results-robot_oob-wide_resnet50_2-fold_2-last/${patient_array[$i]}/Inference-${patient_array[$i]}.csv"

    python frame_visualization.py \
    --title_name "tmp_New_Dataset_Inference-FOLD_1" \
    --sub_title_name ${patient_array[$i]} \
    --GT_path $model_inf_path_1 \
    --model_name "wide_resnet50_2" "resnext50_32x4d" "mobilenet_v3_small" "squeezenet1_0" \
    --model_infernce_path $model_inf_path_1 $model_inf_path_2 $model_inf_path_3 $model_inf_path_4 \
    --results_save_dir "./visual_results/temp"
done

# --filter "median" --kernel_size 19