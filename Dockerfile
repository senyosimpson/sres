FROM python:3.6-alpine
ENV PYTHONBUFFERED 1

COPY sres /
COPY requirements.txt /

RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["sh train.sh"]