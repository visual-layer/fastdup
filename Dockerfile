# syntax=docker/dockerfile:1
FROM ubuntu:20.04
RUN apt update
RUN apt -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt -y install python3.8
RUN apt -y install libopencv-dev libgl1
RUN apt -y install python3-pip
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install fastdup matplotlib matplotlib-inline torchvision pillow pyyaml
