FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

<<<<<<< HEAD:Dockerfile
RUN apt-get update && apt-get install -y \ 
    vim \
    git \
    ffmpeg

# set non iteratctive when installed python-opencv, tzdate
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y --no-install-recommends python-opencv

 # for setup time zone
RUN apt-get install -y tzdata
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata

=======
>>>>>>> upstream/develop:shared_OOB_Inference_Module/Dockerfile
ADD . /OOB_RECOG
WORKDIR /OOB_RECOG

RUN pip install -r requirements.txt
