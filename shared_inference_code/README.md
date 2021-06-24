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
- Robot 40개 [35 / 5] [train / validation]
- Lapa 40개 [35 / 5] [train / validation]
- Robot + Lapa [70 / 10] [train / validation]
- V2 실험부터 100개 dataset으로 작업 예정

--- 
## Development Log
- 2021/06/24 | @jihyun98hutom
    1. Create init code
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
# ex) sudo docker build -t pyl:1.0 .

# docker container running
sudo docker -it -d -v <공유할 폴더 위치>:<컨테이너 내부 공유폴더 위치> --gpus all --name <원하는 컨테이너 이름> <실행하길 원하는 도커 이미지>
# ex) sudo docker -it -d /nas/bgpark/CAM_IO:/data/CAM_IO --gpus all --name pyl-test pyl:1.0

# docker 내부로 진입
# bash shell 실행
sudo docker exec -it pyl-test /bin/bash
```
--- 
## Easy Command
- docker container 생성 (gpu all, volumn 연동, 포트포워딩, ipc 설정)
```shell
docker run -it --name cam_io_hyeongyu -v /home/hyeongyuc/code/CAM_IO:/CAM_IO -v /nas/bgpark:/dat —-gpus all --ipc=host cam_io:1.0
```

- GPU 확인
```shell
watch -d -n 0.5 nvidia-smi
```
---
## 사용법
- input : frame single image or video file
- output : inference result (format : list)
```shell
python infer.py --model_path <model_path> --input_path <input_path>

ex) python infer.py --model_path /home/jihyun/OOB/mobilenet_v3_large-fold1/ckpoint_mobilenet_v3_large-fold1-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt --input_path /data/ROBOT/Video/01_G_01_R_00_ch0_00.mp4
```

- model_path : only support .ckpt file
- input_path : only support .jpg, .png, .mp4, .mpeg, .avi extension file