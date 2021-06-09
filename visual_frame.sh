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

for (( i = 0 ; i < ${#patient_array[@]} ; i++ ))
do
    model_inf_path_1="./results-robot-oob-mobilenet_v3_small-fold_1-last-Inference_1/${patient_array[$i]}/Inference-ROBOT-${patient_array[$i]}.csv"

    python visual_frame.py \
    --title_name "Mobilenet_v3_small-Inference Step_1-FOLD_1" \
    --sub_title_name ${patient_array[$i]} \
    --GT_path $model_inf_path_1 \
    --model_name "mobilenet_v3_small" \
    --model_infernce_path $model_inf_path_1 \
    --results_save_dir "./visual_results/mobilenet-0514-inference-1" \
    --INFERENCE_STEP 1 \
    --WINDOW_SIZE 10000 \
    --OVERLAP_SECTION_NUM 3
done

# --filter "median" --kernel_size 19

patient_array=("R_3" \
            "R_4" \
            "R_6" \
            "R_13" \
            "R_18")

for (( i = 0 ; i < ${#patient_array[@]} ; i++ ))
do
    model_inf_path_1="./results-robot-oob-mobilenet_v3_small-fold_2-last-Inference_1/${patient_array[$i]}/Inference-ROBOT-${patient_array[$i]}.csv"

    python visual_frame.py \
    --title_name "Mobilenet_v3_small-Inference Step_1-FOLD_2" \
    --sub_title_name ${patient_array[$i]} \
    --GT_path $model_inf_path_1 \
    --model_name "mobilenet_v3_small" \
    --model_infernce_path $model_inf_path_1 \
    --results_save_dir "./visual_results/mobilenet-0514-inference-1"
    --INFERENCE_STEP 1 \
    --WINDOW_SIZE 10000 \
    --OVERLAP_SECTION_NUM 3
done

patient_array=("R_7" \
            "R_10" \
            "R_19" \
            "R_56" \
            "R_74")

for (( i = 0 ; i < ${#patient_array[@]} ; i++ ))
do
    model_inf_path_1="./results-robot-oob-mobilenet_v3_small-fold_3-last-Inference_1/${patient_array[$i]}/Inference-ROBOT-${patient_array[$i]}.csv"

    python visual_frame.py \
    --title_name "Mobilenet_v3_small-Inference Step_1-FOLD_3" \
    --sub_title_name ${patient_array[$i]} \
    --GT_path $model_inf_path_1 \
    --model_name "mobilenet_v3_small" \
    --model_infernce_path $model_inf_path_1 \
    --results_save_dir "./visual_results/mobilenet-0514-inference-1" \
    --INFERENCE_STEP 1 \
    --WINDOW_SIZE 10000 \
    --OVERLAP_SECTION_NUM 3
done