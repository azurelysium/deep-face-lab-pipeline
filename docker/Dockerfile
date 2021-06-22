FROM conda/miniconda3:latest

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

#
# Install Miniconda in /opt/conda
# https://hub.docker.com/r/kundajelab/cuda-anaconda-base/dockerfile
#

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /app

#
# Install DeepFaceLab dependencies
# https://github.com/nagadit/DeepFaceLab_Linux
#
RUN conda create -y -n deepfacelab -c main python=3.7 cudnn=7.6.5 cudatoolkit=10.1.243
COPY install.sh /app/install.sh
RUN bash -l install.sh

RUN apt-get update --fix-missing && \
    apt-get install -y ffmpeg libsm6 unzip
COPY env.sh /app/DeepFaceLab_Linux/scripts/env.sh
COPY Merger.py /app/DeepFaceLab_Linux/DeepFaceLab/mainscripts/Merger.py
