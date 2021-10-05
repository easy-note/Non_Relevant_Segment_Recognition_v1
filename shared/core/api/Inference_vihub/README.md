# Out of Body Recognition Project

## 프로젝트 개요
VI Hub를 통해 녹화되는 영상에 대해 환자의 몸속 영상인지, 아닌지를 구분하기 위한 프로젝트

해당 프로젝트를 위해 크게 3가지 Step으로 진행

### Task
1. Inbody, Out of Body [Binary Classification] ==> OOB Task
    - 몸속 안 영상인지, 밖 영상인지 구분하는 Task
    - 해당 Model의 목표치는 FP = 0 에 수렴하도록 학습 (즉, Out body를 Positive라 할때, 모델이 Out Body라고 예측했지만 실제로는 Inbody인 경우) 
2. NIR, RGB (Binary Classification) ==> NIR Task
    - NIR (혈관 조영모드) , RGB 인지 구분하는 Task

3. 서로다른 Device로 획득한 영상 (Robot, Lapa) 에 대한 개별모델이 아닌, 동일한 모델사용시 최적의 성능이 보장되는지에 대한 실험
    - 개뱔 Dataset에 Fitting된 모델학습 후 다른 Device영상 Inference 하여 성능비교

### Dataset
- Robot 100개 [80 / 20] [train / validation]

--- 
## Development Log
- 2021/06/24 | @jihyun98hutom
    1. Create init code (VIHUB pro QA test v.1)
- 2021/09/13 | @jihyun98hutom
    1. Update VIHUB pro QA test v.2 
---

## 초기 환경 설정
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
# ex) sudo docker build -t oob:1.0 .

# docker container running
sudo docker -it -d -v <공유할 폴더 위치>:<컨테이너 내부 공유폴더 위치> --gpus all --name <원하는 컨테이너 이름> <실행하길 원하는 도커 이미지>
# ex) sudo docker -it -d /nas/bgpark/OOB_RECOG:/data/OOB_RECOG --gpus all --name pyl-test oob:1.0

# docker 내부로 진입
# bash shell 실행
sudo docker exec -it pyl-test /bin/bash
```
--- 
## Easy Command
- docker container 생성 (gpu all, volumn 연동, 포트포워딩, ipc 설정)
```shell
docker run -it --name oob_inference -v /home/hyeongyuc/code/OOB_RECOG:/OOB_RECOG —-gpus all --ipc=host oob:1.0
```

- GPU 확인
```shell
watch -d -n 0.5 nvidia-smi
```
---
## Notice
Binary Classification
- In Body (0), Out of Body (1)

Video type
- 'ch1' 비디오에 비디오에 대해서 정상 작동.
- 'xx0' 비디오는 codec issue로 아직 inference 결과가 정확하지 않음. 

~~Inference Step~~
- 30fps video 기준 1fps로 Infereence 되도록 30으로 설정 (default) 

---
## 사용법
### VIHUB_pro_QA_v1
- input
    - moel_path (only support .ckpt format) -> str:
    - frame single image or video file (only support .jpg, .png, .mp4, .mpeg, .avi extension file. recommand 'ch1' video)-> str:
- output
    - inference result (format : list) -> List[int]:

```shell
python infer.py --model_path <model_path> --input_path <input_path>

ex) python infer.py --model_path /home/jihyun/OOB/mobilenet_v3_large-fold1/ckpoint_mobilenet_v3_large-fold1-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt --input_path /data/ROBOT/Video/01_G_01_R_00_ch0_00.mp4
```


### VIHUB_pro_QA_v2 (update 21.09.14)
#### refactor: VIHUB pro production
- input
    - model_path (only support .ckpt format) -> str:
    - target folder path
- output
    - inference result -> List[int]: