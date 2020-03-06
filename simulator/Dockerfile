FROM  ubuntu:16.04
MAINTAINER yuan.wen(y.wen@tianrang-inc.com)
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

RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc
#
#RUN apt-get install -y libxerces-c3-dev && \
#    apt-get install -y swig

COPY ./ /simulator
WORKDIR /simulator

ENTRYPOINT ["python3", "interface.py"]