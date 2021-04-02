FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN pip install pytorch-lightning

ADD . /OOB_RECOG
WORKDIR /OOB_RECOG
