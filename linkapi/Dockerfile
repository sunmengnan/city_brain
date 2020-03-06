FROM  ubuntu:16.04

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git mercurial \
    subversion

RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update -y && \
    apt-get install -y build-essential python3.6

RUN apt-get update -y && \
    apt-get install -y python3-pip

RUN pip3 install eve

COPY ./ /link
WORKDIR /link
EXPOSE 5001

ENTRYPOINT ["python3", "app.py"]