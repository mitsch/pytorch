FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as build-stack

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        git build-essential gcc libcurl4-openssl-dev libz-dev 
        #graphviz

# CMake
WORKDIR /root
RUN git clone https://github.com/Kitware/CMake.git CMake && \
    cd CMake && \
    git checkout tags/v3.15.0 && \
    ./bootstrap --system-curl --parallel=8 && \
    make && \
    make install

# Catch2
WORKDIR /root
RUN git clone https://github.com/catchorg/Catch2.git Catch2 && \
    cd Catch2 && \
    git checkout tags/v2.9.1 && \
    mkdir build && \
    cd build && \
    cmake .. -DCATCH_BUILD_TESTING=OFF -DCATCH_INSTALL_DOCS=OFF && \
    cmake --build . && \
    cmake --install .

FROM build-stack as build

# LibTorch20
WORKDIR /root/libtorch20
ADD . .
RUN mkdir build && cd build && cmake .. && cmake --build . && cmake --install .
