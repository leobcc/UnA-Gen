# --------------------------------------------------------------------------------------------------------------------
# The image needs to be run with: $ docker run -it --gpus all --name UnA-Gen -v /data-net/hupba/lbocchi/UnA-Gen_data:/UnA-Gen/data -p 8763:8888 una-gen
# If changes are not being included in the build use $ docker build --no-cache -t una-gen . 
# To re-enter the running container: $ docker exec -it <container_name_or_id> /bin/bash
# To start the container if it is already up but it is not running: $ docker start <container_name_or_id>
# To attach to a running container with a terminal and resume the process: $ docker attach <container_name_or_id>
# To select the gpu to be used: $ export CUDA_VISIBLE_DEVICES=1,2
# To copy the output folder: $ docker cp ec8cafda7c44:/IF3D/outputs/Video/parkinglot /home-net/lbocchi/IF3D_Project
# --------------------------------------------------------------------------------------------------------------------

# Build the container using the CUDA toolkit version 11.1 image as the base image
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Fetch the keys that are missing from the cuda base image 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# For more details on the installation refer to https://saturncloud.io/blog/how-to-use-gpus-inside-docker-containers-resolving-cuda-version-and-torchcudaisavailable-issues/
# Also, refer to https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-1 \
    && rm -rf /var/lib/apt/lists/*

# Install general dependecies
RUN apt-get update && \
    apt-get install -y \
    python3.7 \
    python3-pip \
    build-essential \
    git \
    git-lfs \
    libglfw3 \
    libglib2.0-0 \
    wget \
    unzip
# libglib2.0-0 probably fail due to request of location and timezone

# Upgrade pip (this is necessary otherwise the installation of setuptools gives problems for some versions)
RUN python3 -m pip install --upgrade pip

# Downgrade numpy to  version 1.23.1 to avoid ImportError: cannot import name 'bool' from 'numpy'
RUN python3 -m pip install numpy==1.23.1

# Set environment variables
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Clone the UnA-Gen repository
RUN git clone https://github.com/leobcc/UnA-Gen.git

# Change working directory to the project folder
WORKDIR /UnA-Gen

# Install the cuda compatible version of torch 1.9.1
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt

RUN mkdir supp_repos
RUN wget -O depth_anything.zip https://github.com/LiheYoung/Depth-Anything/archive/main.zip
RUN unzip depth_anything.zip -d supp_repos
RUN mv supp_repos/Depth-Anything-main supp_repos/Depth_Anything_main
RUN rm depth_anything.zip

RUN pip install -r /UnA-Gen/supp_repos/Depth_Anything_main/requirements.txt

# Remains to fix the copy of the smpl folder and smpl_model checkpoints, 
# as well as the midification in the depth_anything dpt file for a path set manually. 
# Also, the antialias option that needs to be removed cause it's missing from a librayry.
# (one option is to include everything in the repo as an adapted modification)