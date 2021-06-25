FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update
RUN apt-get upgrade
RUN apt-get install vim
RUN apt-get install git

ADD . /OOB_RECOG
WORKDIR /OOB_RECOG

RUN apt-get install python3-opencv
RUN pip install -r requirements.txt