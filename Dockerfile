FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \ 
    vim \
    git

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y --no-install-recommends python-opencv

ADD . /OOB_RECOG
WORKDIR /OOB_RECOG

RUN pip install -r requirements.txt