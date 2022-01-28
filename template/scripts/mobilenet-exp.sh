python new_visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 1 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \       
--experiment_sub_type 'semi' \
--n_stage 3 \
--multi_stage \
--use_test_batch \
--semi_data 'rs-general' \
 --save_path '/code/OOB_Recog/logs-semi/mobilenet-rs-general-ws-general'



python new_visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --train_stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--update_type 2 \
--update_type2 True \
--use_test_batch \
 --save_path '/code/OOB_Recog/logs-aaa/tatas'


#########


nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \       
--WS_ratio 3 \
--use_wise_sample \
--experiment_sub_type 'semi' \
--n_stage 3 \
--multi_stage \
--semi_data 'rs-general' \
 --save_path '/code/OOB_Recog/logs-semi/mobilenet-rs-general-ws-general' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '5' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 2 \
--update_type2 True \
--use_step_weight True \
--experiment_sub_type 'semi' \
--n_stage 3 \
--multi_stage \
--semi_data 'rs-general' \
 --save_path '/code/OOB_Recog/logs-semi/mobilenet-rs-general-ws-proxy-base' > /dev/null &





nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--update_type 2 \
--update_type2 True \
--experiment_sub_type 'semi' \
--n_stage 3 \
--multi_stage \
--semi_data 'rs-proxy-base' \
 --save_path '/code/OOB_Recog/logs-semi/mobilenet-rs-proxy-base-rs-general' > /dev/null &


 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 2 \
--update_type2 True \
--use_step_weight True \
--experiment_sub_type 'semi' \
--n_stage 3 \
--multi_stage \
--semi_data 'rs-proxy-base' \
 --save_path '/code/OOB_Recog/logs-semi/mobilenet-rs-proxy-base-rs-proxy-base' > /dev/null &



# Proxy - correct + WS / 0 1 2 3 4 6
 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '6' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
 --save_path '/code/OOB_Recog/logs-mobilenet/mobilenet-rs-general' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '7' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
 --save_path '/code/OOB_Recog/logs-mobilenet/mobilenet-ws-general' > /dev/null &


 nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-rs-general' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '5' \
--IB_ratio 3 --random_seed 3829 --stage 'general_train' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-ws-general' > /dev/null &


 #####
nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 2 \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-rs-proxy-2' > /dev/null &

 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 3 \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-rs-proxy-3' > /dev/null &

 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 4 \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-rs-proxy-4' > /dev/null &

 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '6' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 5 \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-rs-proxy-5' > /dev/null &

######
 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 2 \
--update_type2 True \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-ws-proxy-type2-2' > /dev/null &

 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 3 \
--update_type2 True \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-ws-proxy-type2-3' > /dev/null &

 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 4 \
--update_type2 True \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-ws-proxy-type2-4' > /dev/null &

 
 ######

nohup python visual_flow.py --fold '1' --trial 1 --model "mobile_vit" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 1 \
 --save_path '/code/OOB_Recog/logs-vit/mvit-ws-online-method-1-again' > /dev/null &


############



 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 2 \
--update_type2 False \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-rs-proxy-type1-2-step' > /dev/null &


 nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 3 \
--update_type2 False \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-rs-proxy-type1-3-step' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "mobilenetv3_large_100" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '2' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 4 \
--update_type2 False \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-proxy-update-exp/mobilenet-rs-proxy-type1-4-step' > /dev/null &



##############


 nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '0' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 2 \
--update_type2 False \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-rs-proxy-type1-2-step' > /dev/null &


 nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '1' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 4 \
--update_type2 False \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-rs-proxy-type1-4-step' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 2 \
--update_type2 True \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-rs-proxy-type2-2-step' > /dev/null &


 nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--update_type 4 \
--update_type2 True \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-rs-proxy-type2-4-step' > /dev/null &



 nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '3' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 2 \
--update_type2 False \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-ws-proxy-type1-2-step' > /dev/null &


 nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '4' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 4 \
--update_type2 False \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-ws-proxy-type1-4-step' > /dev/null &


nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '6' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 2 \
--update_type2 True \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-ws-proxy-type2-2-step' > /dev/null &

 nohup python visual_flow.py --fold '1' --trial 1 --model "resnet18" --max_epoch 100 \
--batch_size 256 --lr_scheduler "step_lr" --lr_scheduler_step 5 --lr_scheduler_factor 0.9 --cuda_list '6' \
--IB_ratio 3 --random_seed 3829 --stage 'hem_train' --hem_extract_mode 'hem-emb-online' --inference_fold '1' \
--WS_ratio 3 \
--use_wise_sample \
--update_type 4 \
--update_type2 True \
--use_step_weight True \
 --save_path '/code/OOB_Recog/logs-resnet/resnet-ws-proxy-type2-4-step' > /dev/null &