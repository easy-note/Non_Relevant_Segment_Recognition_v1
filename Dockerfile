FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \ 
    vim \
    git

ENV DEBIAN_FRONTEND=noninteractive # set non iteratctive when installed python-opencv, tzdate
RUN apt-get install -y --no-install-recommends python-opencv

RUN apt-get install -y tzdata # for setup time zone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata

ADD . /OOB_RECOG
WORKDIR /OOB_RECOG

RUN pip install -r requirements.txt