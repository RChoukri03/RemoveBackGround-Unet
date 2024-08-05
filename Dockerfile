FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install -r requirements.txt


WORKDIR /opt/work