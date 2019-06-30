FROM ubuntu
ENV PYTHONBUFFERED 1

COPY . /
COPY requirements.txt /

RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["sh train.sh"]