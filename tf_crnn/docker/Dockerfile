FROM tensorflow/tensorflow:1.8.0-gpu

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    CUDA_VISIBLE_DEVICES=0

# update apt sources
COPY docker/sources.list /etc/apt/sources.list

# install for opencv lib
RUN apt-get update -y && \
    apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender1 \
    git

COPY docker/requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt -i https://mirrors.aliyun.com/pypi/simple && \
    rm -rf /tmp/*

# install sshd
RUN apt-get update && \
    apt-get install -y \
    sudo vim openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    # SSH login fix. Otherwise user is kicked off after login
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

WORKDIR /app
COPY ./ /app

EXPOSE 22
ENV PROJECT_DIR=/app

CMD ["/usr/sbin/sshd", "-D"]
