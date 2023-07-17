FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6
# upgrade pip
RUN pip install --upgrade pip

# Installing reruirements
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install -r requirements-pip.txt

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/