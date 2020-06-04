FROM nvidia/cuda:10.0-devel-ubuntu18.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=2.0.0b1
ENV PYTORCH_VERSION=1.3.0
ENV TORCHVISION_VERSION=0.4.1
ENV CUDNN_VERSION=7.6.0.64-1+cuda10.0
ENV NCCL_VERSION=2.4.8-1+cuda10.0
ENV MXNET_VERSION=1.5.0
ARG python=3.6
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


# update apt
RUN apt-get update

# install general dependencies
RUN apt-get install -y python3-pip wget git autoconf libtool nvidia-modprobe \
    software-properties-common cmake openssh-server openssh-client numactl

# setup cluster user and SSH access to container
ENV USER cluster
RUN useradd -ms /bin/bash $USER && usermod -p '*' $USER
ENV HOME /home/$USER
ENV SSHDIR $HOME/.ssh
RUN mkdir -p ${SSHDIR} \
    && touch ${SSHDIR}/sshd_config \
    && ssh-keygen -t rsa -f ${SSHDIR}/ssh_host_rsa_key -N '' \
    && cp ${SSHDIR}/ssh_host_rsa_key.pub ${SSHDIR}/authorized_keys \
    && cp ${SSHDIR}/ssh_host_rsa_key ${SSHDIR}/id_rsa \
    && echo "    IdentityFile ${SSHDIR}/id_rsa" >> ${SSHDIR}/config \
    && echo "    StrictHostKeyChecking no" >> ${SSHDIR}/config \
    && echo "    UserKnownHostsFile /dev/null" >> ${SSHDIR}/config \
    && echo "    Port 2022" >> ${SSHDIR}/config \
    && echo 'Port 2022' >> ${SSHDIR}/sshd_config \
    && echo "HostKey ${SSHDIR}/ssh_host_rsa_key" >> ${SSHDIR}/sshd_config \
    && echo "PidFile ${SSHDIR}/sshd.pid" >> ${SSHDIR}/sshd_config \
    && echo "PasswordAuthentication no" >> ${SSHDIR}/sshd_config \
    && chmod -R 600 ${SSHDIR}/* \
    && chown -R ${USER}:${USER} ${SSHDIR}/

# set path for ssh user
RUN echo "export PATH=\"/usr/local/cuda/bin:\$PATH\"" > $HOME/.bashrc

# install library packages
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py

# upgrade packages
RUN pip3 install -U setuptools


# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip3 install future typing
RUN pip3 install numpy \
        tensorflow-gpu==${TENSORFLOW_VERSION} \
        keras \
        h5py

RUN pip3 install https://download.pytorch.org/whl/cu100/torch-${PYTORCH_VERSION}%2Bcu100-$(python -c "import wheel.pep425tags as w; print('-'.join(w.get_supported()[0][:-1]))")-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu100/torchvision-${TORCHVISION_VERSION}%2Bcu100-$(python -c "import wheel.pep425tags as w; print('-'.join(w.get_supported()[0][:-1]))")-linux_x86_64.whl

# Install tqdm, tenboardX
RUN pip3 install tqdm
RUN pip3 install tensorboardX

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.2.tar.gz && \
    tar zxf openmpi-4.0.2.tar.gz && \
    cd openmpi-4.0.2 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi


# Install modified horovod
RUN cd ${HOME} && \
    wget https://dt-training.s3.amazonaws.com/horovod-modified.tar.gz && \
    tar -zxf horovod-modified.tar.gz && \
    cd ${HOME}/horovod-modified && \
    pwd && \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_CUDA_HOME=/usr/local/cuda \
    HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/compat \
    python setup.py install

# Install mimic distributed training program
RUN cd ${HOME} && \
    git clone https://github.com/zarzen/mimic_dt.git && \
    cd ${HOME}/mimic_dt && \
    make MPI=1 CUDA_HOME=/usr/local/cuda


COPY container_entrypoint.sh /etc/
ENTRYPOINT /etc/container_entrypoint.sh

EXPOSE 2022