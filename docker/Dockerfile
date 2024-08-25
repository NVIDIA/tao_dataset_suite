FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6

# upgrade pip
RUN pip install --upgrade pip

# Installing requirements
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install -r requirements-pip.txt && \
    rm -rf requirements-pip.txt

# Installing TAO-Core
# Installing TAO-Core
COPY tao-core tao-core
RUN cd tao-core && bash release/python/build_wheel.sh && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    rm -rf *

# Installing PyTorch
COPY tlt-pytorch tlt-pytorch
RUN export WORKSPACE=/workspace/tlt-pytorch && \
    cd ${WORKSPACE} && bash release/docker/build_wheel.sh && \
    find dist/ -name "nvidia_tao_pytorch*.whl" -type f | xargs -n 1 pip install && \
    rm -rf *

# Install mmcv from source for our cuda versions.
WORKDIR /opt
COPY third_party/mmcv/mmcv.patch mmcv.patch
RUN git clone https://github.com/open-mmlab/mmcv.git \
  && cd mmcv && git checkout v1.7.1 \
  && git apply /opt/mmcv.patch \
  && pip install -r requirements/optional.txt --ignore-installed \
  && FORCE_CUDA=1 MMCV_WITH_OPS=1 python setup.py install

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/