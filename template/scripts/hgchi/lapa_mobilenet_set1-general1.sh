# 22-01-28 baby model학습시키기

# offline 용 visual_flow (baby model 학습시키기용도) - baby model 학습할때의 기준으로 생각해서 셋팅하기.
## hem_iteration_idx가 100(초기) 이 외에는 appointment_assets_path 에서 불러옴
## appointment_assets_path는 args긴 한데, visual scripts 에서는 사용하게 할수도 있고, 사용하지 않게 할수도 있을 것 같음.
## visual flow에서는 appointment_assets_path가 baby_model 학습시킬때 사용하는 csv (즉, 이전 stage의 hem model 학습할때 사용한 csv)
## sripts 에 args작성해서 사용하게 하는 경우는 csv를 사용자가 지정해주도록 하는 방식
## script 에 args작성 못하게 하는 경우는 apply_offline_flow처럼 내부적으로 알아서 nas에서 가져오게하는 방식.

## 일단은 args에 작성못하게하는 구조로 작성함. (model기준으로 sota setting으로 가져오도록 고정시켜버림 => ib, ws ratio 3,  softmax_diff_small, n_dropout = 5, top_ratio = 0.5(?))
# ==> hem_extract_mode의 의미 : 불러올 hem assets csv 를 뽑기위해 사용한 methods?
#### hem_iteration_idx == 100 일때는 아무의미 없음
#### 이외에는 csv 불러와야 하기 때문에 의미 부여가능, (model 도 동일한 의미로 사용 ==> 이중의미 (실제로 baby model학습모델 + 이전 iteration 에서 model을 사용해서 뽑은 hem_assets csv))
# ==> 이렇게 따져야 iteration 이기 때문에 이중의미를 갖는게 우선은 맞아보임

## 불러올 csv 확장할떄는, model기준 hem_extract_mode별로 sota setting 만 iteration 할수있도록 추가확장 셋팅하면 될 듯.

# --inference 부분은 없어도 됨, 어차피 안함.



IB_ratio=3;

top_ratio=(0.05);
WS_ratio=3;
n_dropout=5;

hem_interation_idx=100;

for ratio in "${top_ratio[@]}";
do
    nohup python -u ../new_visual_flow.py \
        --hem_interation_idx ${hem_interation_idx}\
        --use_wise_sample \
        --dataset "LAPA" \
        --fold "1" \
        --WS_ratio ${WS_ratio} \
        --model "mobilenetv3_large_100" \
        --pretrained \
        --use_lightning_style_save \
        --max_epoch 100 \
        --batch_size 256 \
        --lr_scheduler "step_lr" \
        --lr_scheduler_step 5 \
        --lr_scheduler_factor 0.9 \
        --cuda_list "4" \
        --random_seed 3829 \
        --IB_ratio ${IB_ratio} \
        --train_stage "general_train" \
        --inference_fold "1" \
        --inference_interval "30" \
        --hem_extract_mode "hem-softmax_diff_small-offline" \
        --experiments_sheet_dir "/OOB_RECOG/results-lapa/mobilenet-set1-general1" \
        --save_path "/OOB_RECOG/logs-lapa/mobilenet-set1-general1" > "lapa-set1-general1.out"
done;