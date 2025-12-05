FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 9.0a 10.0 11.0 12.0+PTX"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# upgrade pip
RUN pip install --upgrade pip

# Remove any existing opencv installations to avoid conflicts
RUN pip uninstall -y opencv || true && \
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2

# Installing requirements
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install -r requirements-pip.txt && \
    rm -rf requirements-pip.txt

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