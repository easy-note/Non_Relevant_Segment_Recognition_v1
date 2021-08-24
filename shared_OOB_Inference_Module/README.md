# Vi-HUB Pro (Out of Body Recognition Project)

## 프로젝트 개요
VI Hub를 통해 녹화되는 영상에 대해 환자의 몸속 영상인지, 아닌지를 구분하기 위한 프로젝트

해당 프로젝트를 위해 크게 3가지 Step으로 진행

### Task
1. Inbody, Out of Body [Binary Classification] ==> OOB Task
    - 몸속 안 영상인지, 밖 영상인지 구분하는 Task
    - 해당 Model의 목표치는 FP = 0 에 수렴하도록 학습 (즉, Out body를 Positive라 할때, 모델이 Out Body라고 예측했지만 실제로는 Inbody인 경우) 

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
- 2021/08/10 | @jihyun98hutom
    1. Create init code
---

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
## Input constraints
### Model Predict Output file - Input CSV example
- Must be in csv file format.
    - Inference-ROBOT-01_G_01_R_100_ch1_01.csv
```
,predict
0,0
1,0
2,0
3,0
4,0
5,0
6,0
7,0
8,0
9,0
10,0
11,0
12,0
13,0
14,0
15,0
16,1
17,1
18,1
19,1
20,1
21,0
```
### Annotation file - Input JSON example
- Must be in json file format.
    - 01_G_01_R_100_ch1_01_OOB_27.json
```
{
  "totalFrame": 137424,
  "frameRate": 30,
  "width": 1280,
  "height": 1024,
  "_id": "60ca804b06c8f9001b311784",
  "annotations": [
    {
      "start": 23522,
      "end": 24393,
      "code": 1
    },
    {
      "start": 49214,
      "end": 49721,
      "code": 1
    },
    {
      "start": 101265,
      "end": 101795,
      "code": 1
    },
    {
      "start": 119824,
      "end": 120382,
      "code": 1
    }
  ],
  "annotationType": "OOB",
  "createdAt": "2021-06-16T22:50:51.218Z",
  "updatedAt": "2021-06-21T08:00:45.394Z",
  "annotator": "27",
  "name": "01_G_01_R_100_ch1_01",
  "label": {
    "1": "OutOfBody"
  }
}
```

---
## How to Use
Required arguments:
```
--model_output_csv_path               model predict output file path
--gt_json_path                        ground-truth (annotation) file path
--inference_step                      inference frame step
```
Example
```shell
python OOB_inference_module.py --model_output_csv_path <model_output_csv_path> --gt_json_path <gt_json_path> --inference_step <inference_step>

>>> python OOB_inference_module.py --model_output_csv_path '/OOB_RECOG/shared_OOB_Inference_Module/assets/Inference/Inference-ROBOT-01_G_01_R_100_ch1_01.csv' --gt_json_path '/OOB_RECOG/shared_OOB_Inference_Module/assets/Annotation(V2)/01_G_01_R_100_ch1_01_OOB_27.json' --inference_step 5
```

Output file
- `results/results_OR_CR.json`
```
{
  "01_G_01_R_100_ch1_01": {
    "over_estimation_ratio": 0.18985270049099837,
    "confidence_ratio": 0.6202945990180033
  }
}
``` 

