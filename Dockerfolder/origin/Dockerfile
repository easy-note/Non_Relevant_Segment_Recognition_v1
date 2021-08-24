FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# RUN pip install pytorch-lightning

# ADD . /OOB_RECOG
# WORKDIR /OOB_RECOG

# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# RUN apt-get update && apt-get install -y \ 
#     vim \
    # git
    # ffmpeg
    
# set non iteratctive when installed python-opencv, tzdate
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y --no-install-recommends python-opencv

 # for setup time zone
RUN apt-get install -y tzdata
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata

ADD . /OOB_RECOG
WORKDIR /OOB_RECOG

# RUN pip install -r requirements.txt
