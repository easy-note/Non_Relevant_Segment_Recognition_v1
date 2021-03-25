FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN pip install pytorch-lightning

ADD . /CAM_IO
WORKDIR /CAM_IO
