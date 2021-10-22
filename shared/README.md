# Vi-HUB Pro (Out of Body Recognition Project)

## 프로젝트 개요
VI Hub를 통해 녹화되는 영상에 대해 환자의 몸속 영상인지, 아닌지를 구분하기 위한 프로젝트

해당 프로젝트를 위해 크게 3가지 Step으로 진행

### Task
1. Inbody, Out of Body [Binary Classification] ==> OOB Task
    - 몸속 안 영상인지, 밖 영상인지 구분하는 Task
    - 해당 Model의 목표치는 FP = 0 에 수렴하도록 학습 (즉, Out body를 Positive라 할때, 모델이 Out Body라고 예측했지만 실제로는 Inbody인 경우) 
2. NIR, RGB (Binary Classification) ==> NIR Task
    - NIR (혈관 조영모드) , RGB 인지 구분하는 Task

3. 서로 다른 Device로 획득한 영상 (Robot, Lapa) 에 대한 개별모델이 아닌, 동일한 모델사용시 최적의 성능이 보장되는지에 대한 실험
    - 개뱔 Dataset에 Fitting된 모델학습 후 다른 Device영상 Inference 하여 성능비교

### Dataset
- Robot 100개 [80 / 20] [train / validation]
    - validation dataset (fold 1)
        - R_2, R_6, R_13, R_74, R_100, R_202, R_301, R_302, R_311, R_312, R_313, R_336, R_362, R_363, R_386, R_405, R_418, R_423, R_424, R_526
    
---
## Notice
Class
- Binary Classification
  - In Body (0), Out of Body (1)

Estimation Metrics
- Over Estimation Ratio
  - `FP / (FP+TP+FN)`
- Confidence Ratio
  - `(TP-FP) / (FP+TP+FN)`


--- 
## Development Log
- 2021/06/24 | @jihyun98hutom
    1. Create init code (VIHUB pro QA test v.1)
- 2021/08/10 | @jihyun98hutom
    1. Create init code (Evaluation-vihub)
- 2021/09/13 | @jihyun98hutom
    1. Update VIHUB pro QA test v.2 
- 2021/10/05 | @hyeongyuc96hutom
    1. Evaluation-vihub: gt length와 predict length가 서로 맞지 않을 경우 동일하게 맞추어 metric 계산할 수 있도록 도와주는 evaluation_Helper 추가
- 2021/10/05 | @hyeongyuc96hutom
    1. 기존 분리된 shared_inference/evaluation module 새로운 hireachy로 정리 (/shared/core/api)
    2. 기존 분리된 shared_inference/evaluation module Dockerfile 및 requirements file 통합 (/shared/env)
    3. vihub-pro module flow와 동일하게 linux 환경에서 Inference/Evaluation 할 수 있도록 도와주는 utils module 및 sciprt 작성 (/shared/core/utils, /shared/script)
    4. OOB test model upload (/shared/model/mobilenet_v3_large.ckpt)
- 2021/10/14 | @hyeongyuc96hutom
    1. evaluation module return 형식 변경, json file을 저장하지 않고 json format string return으로 변경
---

## Mobule/File Path 
1. Enviroments
```bash
# Env
/shared/env/Dockerfile
/shared/env/requirements.txt
```
2. [Inference](./core/api/Inference_vihub)
```bash
# module path
/shared/core/api/Inference_vihub
```
3. [Evaluation](./core/api/Evaluation_vihub)
```bash
# module path
/shared/core/api/Evaluation_vihub
```
4. [Utils](./core/api/utils)
```bash
# utils path
/shared/core/api/utils/ffmpegHelper # vihub pro와 실험환경 등에서 사용하는 ffmpeg을 통합한 module [frame cutting, frame langth, fps..]
/shared/core/api/utils/evalHelper # predict list length가 gt json length와 동일하지 않을 경우 동일하게 맞추어 다시 predict/gt list return을 도와주는 module
/shred/core/api/utils/visualHelper # predict(mobile, efficient) 와 gt의 FP, FN frame visiaulzation 을 도와주는 module  
```
4. [Script](./core/script)
```bash
# script path
/shared/script/inference.py[.sh] # linux 환경에서 vihub pro flow와 동일하게 1fps로 video cutting 후 해당 frame에 대해 evaluation 및 visualziation script
/shared/script/evaluation.py[.sh] # predict list length가 gt json length와 동일하지 않을 경우 동일하게 맞추어 evaluation 및 visualization script 
```
5. Model(./model)
```bash
# model
/shared/model
```

## Module Flow
(각 module 사용법은 해당 module dir의 README 참조)
1. Enviroment setup
- Dockerfile/requirements.txt
2. call Inference module
- input
    - model_path (only support .ckpt format) -> str:
    - target folder path
- output
    - inference result -> List[int]:
3. call Evaluation module
- input
    - model_output_csv_path -> str(.csv)
    - gt_json_path: ground-truth -> str(.json)
    - inference_step: inference interval -> int
- output
    -  Evaluation result -> file(.json):
4. check resuts / predict list or metric (CR, OR)


## Init Environment Setting
```bash
# nvidia-docker repository 등록
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# package list update & install
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 간단 test
docker run --runtime=nvidia --rm nvidia/cuda:10.0-base nvidia-smi
```

docker build & run

```jsx
# Dockerfile이 있는 곳에서
# docker image building
sudo docker build -t <원하는 도커 이미지 이름>:<버전 정보 원하면> .
# ex) sudo docker build -t oob_vihub_mobule:1.0 .

# docker container running
sudo docker -it -d -v <공유할 폴더 위치>:<컨테이너 내부 공유폴더 위치> --gpus all --name <원하는 컨테이너 이름> <실행하길 원하는 도커 이미지>
# ex) sudo docker -it -d /nas/bgpark/OOB_RECOG:/data/OOB_RECOG --gpus all --name pyl-test oob_vihub_mobule:1.0

# docker 내부로 진입
# bash shell 실행
sudo docker exec -it pyl-test /bin/bash
```
--- 
## Easy Command
- docker container 생성 (gpu all, volumn 연동, 포트포워딩, ipc 설정)
```shell
docker run -it --name vihub_pro_module_test -v /home/hyeongyuc/code/OOB_RECOG/shared:/OOB_RECOG —-gpus all --ipc=host oob_vihub_mobule:1.0
```

- GPU 확인
```shell
watch -d -n 0.5 nvidia-smi
```