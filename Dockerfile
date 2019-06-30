FROM ubuntu
ENV PYTHONBUFFERED 1

COPY . /
COPY requirements.txt /

RUN apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["sh train.sh"]