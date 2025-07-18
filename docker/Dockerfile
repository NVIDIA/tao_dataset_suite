FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6

# upgrade pip
RUN pip install --upgrade pip

# Installing requirements
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install -r requirements-pip.txt && \
    rm -rf requirements-pip.txt

# 25.01 DLFW stack includes 12.0+PTX which throws an error when building our custom CUDA kernels
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0"

# Installing PyTorch
COPY tao-pytorch tao-pytorch
RUN export WORKSPACE=/workspace/tao-pytorch && \
    cd ${WORKSPACE} && bash release/docker/build_wheel.sh && \
    find dist/ -name "nvidia_tao_pytorch*.whl" -type f | xargs -n 1 pip install && \
    rm -rf *

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/