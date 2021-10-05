# set torch cuda env
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# set non iteratctive when installed tzdate
ENV DEBIAN_FRONTEND=noninteractive

# for setup time zone
RUN apt-get install -y tzdata
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata

# set workdir and cp code 
ADD ../shared/ /OOB_RECOG
WORKDIR /OOB_RECOG

# download requirements
RUN pip install -r requirements.txt