# A base container for training on Paperspace Gradient
FROM nvidia/cuda:10.0-base-ubuntu18.04
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /

RUN apt update && apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]
