FROM nvidia/cuda:10.0-cudnn7-runtime

#build時に実行されるコード
RUN apt update && apt install -y python3-pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev
WORKDIR /
ADD requirements.txt /
RUN pip3 install -r requirements.txt
RUN apt install -y wget