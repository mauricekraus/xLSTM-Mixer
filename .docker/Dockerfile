# syntax = docker/dockerfile:1.2
ARG KIND=3.11
FROM mkrausio/ml_research:base-3.11-cuda-12.1.0

ARG COMPOSE_PROJ_NAME="xlstm_mixer"
ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}

RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    unzip \
    git-lfs \
    ninja-build \
    python3.11-dev \
    wget \
    cmake \
    g++-11 \
    libmkl-dev \
    && apt-get clean

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN ldconfig /usr/local/cuda-12.1/compat/

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:$PATH

RUN mkdir -p /workspaces/$PROJECT_NAME
WORKDIR /workspaces/$PROJECT_NAME

COPY requirements.txt .
COPY lightning_requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install torch==2.3.1 && \
    pip install -r requirements.txt && \
    pip install -r lightning_requirements.txt || echo "pip resolved failed lightning_requirements.txt, but seems to be fine"


ENV PYTHONPATH=$PYTHONPATH:src

CMD zsh