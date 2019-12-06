FROM nvidia/cuda:10.1-runtime-ubuntu18.04

# System packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        less \
        sudo \
        mc \
        screen

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh \
    && bash Miniconda3-4.7.12-Linux-x86_64.sh -p /miniconda3 -b \
    && rm Miniconda3-4.7.12-Linux-x86_64.sh

ENV PATH=/miniconda3/bin:${PATH}

RUN pip install \
    torch==1.3.1 \
    torchvision \
    jupyter \
    numpy \
    pyyaml \
    pandas \
    nltk


###############################################################################
# Adding user with same priviliges as host user. With free access to sudo group
###############################################################################
ARG USER_NAME
ARG GID_NUMBER
ARG UID_NUMBER
RUN groupadd -g $GID_NUMBER $USER_NAME \
    && useradd \
    --create-home \
    --no-log-init \
    --uid $UID_NUMBER \
    --gid $GID_NUMBER \
    $USER_NAME \
    && adduser $USER_NAME sudo \
    && passwd -d $USER_NAME

USER $USER_NAME
