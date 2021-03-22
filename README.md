# CAM_IO
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
