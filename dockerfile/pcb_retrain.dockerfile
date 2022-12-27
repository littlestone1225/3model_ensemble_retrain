# Tesla P4
# FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# A5000
# FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

ARG USER=adev
ARG UID=1000
ARG GID=1000
ARG AOI_DIR_NAME=""

ENV DISPLAY :11
ENV DEBIAN_FRONTEND noninteractive

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# https://stackoverflow.com/questions/56131677/run-pip-install-there-was-a-problem-confirming-the-ssl-certificate-ssl-certi
# pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org

# Install apt package
RUN apt-get update && \
    apt-get install -y sudo vim git wget curl zip unzip dmidecode && \
    apt-get install -y net-tools iputils-ping apt-utils && \
    # For opencv-python
    # (pkg-config --cflags opencv)
    # (pkg-config --modversion opencv)
    apt-get install -y libsm6 libxrender-dev libopencv-dev && \
    # Install make
    apt-get install -y build-essential

RUN apt-get install -y software-properties-common && \
    cd /usr/bin && \
    ln -s python3.6 python && \
    apt-get install -y python3-pip python3.6-dev && \
    ln -s pip3 pip && \
    python3 -m pip install protobuf==3.19.4 --user


# Install python3.6 package
RUN apt install unixodbc-dev --yes && \
    pip3 install afs2-datasource afs2-model && \
    pip3 install opencv-python==4.1.1.26 && \
    pip3 install psutil && \
    pip3 install filelock==3.0.12 && \
    pip3 install SharedArray==3.2.1 && \
    pip3 install termcolor==1.1.0 && \
    #pip3 install protobuf==4.21.1 && \
    pip3 install Cython==0.29.21 && \
    pip3 install matplotlib && \
    pip3 install scipy==1.5.4 && \
    pip3 install PyYAML==3.11 && \
    pip3 install pycocotools==2.0.2


# CUDA 10.2
# RUN pip3 install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

# CUDA 11.0
# RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.1
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.3
# RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Set the home directory to our user's home.
ENV USER=$USER
ENV HOME="/home/$USER"
ENV AOI_DIR_NAME=$AOI_DIR_NAME

RUN echo "Create $USER account" &&\
    # Create the home directory for the new $USER
    mkdir -p $HOME &&\
    # Create an $USER so our program doesn't run as root.
    groupadd -r -g $GID $USER &&\
    useradd -r -g $USER -G sudo -u $UID -d $HOME -s /sbin/nologin -c "Docker image user" $USER &&\
    # Set root user no password
    mkdir -p /etc/sudoers.d &&\
    echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER && \
    # Chown all the files to the $USER
    chown -R $USER:$USER $HOME

# Change to the $USER
WORKDIR $HOME
USER $USER
