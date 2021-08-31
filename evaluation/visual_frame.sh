: << "END"
save results file name = {--title_name}-{--sub_title_name}.png

--model_name (yticks name)

--model_inference_path 
these args option input should be pair with ordering
if not, results are not synced.

--INFERENCE_STEP : Inference step as same as in test.py, if unmatced it will be error
--WINDOW_SIZE : included frame per section on calc metric
--OVERLAP_SECTION_NUM : overlap window size, 1 is non-overlap 

--filter [median, opening, closing] 

--kernel_size [1,2,3,4,5,6,7,9,11,12,13,15,17,18,19,21,35,29,45,59], default = 1
only apply in predict results, not GT
if you dont want to apply filter, remove --filter and --kernel_size
you can also composite sequence of filter like above example, but you shuold notice that --fitler, --filter_size must be sync 
--filter "median" "opening"
--filter_size 5 3

--ignore_kernel_size
this option is used in consistency filter (not yet support)
END

patient_array=("R_2" \
            "R_6" \
            "R_13" \
            "R_74" \
            "R_100" \
            "R_202" \
            "R_301" \
            "R_302" \
            "R_311" \
            "R_312" \
            "R_313" \
            "R_336" \
            "R_362" \
            "R_363" \
            "R_386" \
            "R_405" \
            "R_418" \
            "R_423" \
            "R_424" \
            "R_526")

opening_filter_size_array=(3)

closing_filter_size_array=(18)

median_filter_size_array=(11)

# when you want to visualize for only single model / and applid single filter. kernel
: << "END"
for (( i = 0 ; i < ${#patient_array[@]} ; i++ ))
do
    for (( j = 0 ; j < ${#median_filter_size_array[@]} ; j++ ))
    do
        model_inf_path_1="./results_v2_new_robot_oob-mobilenet_v3_large-fold_1-last-1fps/${patient_array[$i]}/Inference-ROBOT-${patient_array[$i]}.csv"

        python visual_frame.py \
        --title_name "mobilenet_v3_large-InferenceStep_30" \
        --sub_title_name ${patient_array[$i]} \
        --GT_path $model_inf_path_1 \
        --model_name "mobilenet_v3_large" \
        --model_infernce_path $model_inf_path_1 \
        --results_save_dir "./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/median-${median_filter_size_array[$j]}/${patient_array[$i]}" \
        --INFERENCE_STEP 30 \
        --WINDOW_SIZE 1000 \
        --OVERLAP_SECTION_NUM 2 \
        --filter "median" --kernel_size ${median_filter_size_array[$j]}
    done
done
END


# when you want to compare multi model 
: << "END"
for (( i = 0 ; i < ${#patient_array[@]} ; i++ ))
do
    model_inf_path_1="./results_v2_new_robot_oob-mobilenet_v3_large-fold_1-last-1fps/${patient_array[$i]}/Inference-ROBOT-${patient_array[$i]}.csv"
    model_inf_path_2="./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/median-1/${patient_array[$i]}/mobilenet_v3_large-InferenceStep_30-${patient_array[$i]}-Inference-post.csv"
    model_inf_path_3="./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/median-3/${patient_array[$i]}/mobilenet_v3_large-InferenceStep_30-${patient_array[$i]}-Inference-post.csv"
    model_inf_path_4="./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/median-5/${patient_array[$i]}/mobilenet_v3_large-InferenceStep_30-${patient_array[$i]}-Inference-post.csv"
    model_inf_path_5="./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/median-7/${patient_array[$i]}/mobilenet_v3_large-InferenceStep_30-${patient_array[$i]}-Inference-post.csv"
    model_inf_path_6="./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/median-9/${patient_array[$i]}/mobilenet_v3_large-InferenceStep_30-${patient_array[$i]}-Inference-post.csv"
    model_inf_path_7="./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/median-11/${patient_array[$i]}/mobilenet_v3_large-InferenceStep_30-${patient_array[$i]}-Inference-post.csv"

    python visual_frame.py \
    --title_name "mobilenet_v3_large-InferenceStep_30" \
    --sub_title_name ${patient_array[$i]} \
    --GT_path $model_inf_path_1 \
    --model_name "mobilenet_v3_large" "median-1"  "median-3"  "median-5"  "median-7"  "median-9" "median-11" \
    --model_infernce_path $model_inf_path_1 $model_inf_path_2 $model_inf_path_3 $model_inf_path_4 $model_inf_path_5 $model_inf_path_6 $model_inf_path_7 \
    --results_save_dir "./POST_PROCESSING/robot-oob-v2-mobilenet_v3_large-1fps/total-median/${patient_array[$i]}" \
    --INFERENCE_STEP 30 \
    --WINDOW_SIZE 1000 \
    --OVERLAP_SECTION_NUM 2
done
END