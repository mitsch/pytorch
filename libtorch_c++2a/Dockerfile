FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as build-stack

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        git build-essential gcc libcurl4-openssl-dev libz-dev wget curl
        #graphviz

# GCC
WORKDIR /root
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        bison flex 
RUN git clone https://github.com/gcc-mirror/gcc.git gcc && \
    cd gcc && \
    git checkout 6bda7a942a0ed71678622d2a4f8869557b3c38f7
    # git checkout gcc-9_2_0-release
RUN cd gcc && \
    sed -i contrib/download_prerequisites -e '/base_url=/s/ftp/http/' && \
    bash contrib/download_prerequisites && \
    cd .. && \
    mkdir gcc-build && \
    cd gcc-build && \
    ../gcc/configure --disable-multilib --enable-languages=c,c++ && \
    make -j 10 && \
    make install && \
    echo "/usr/local/lib64" > /etc/ld.so.conf.d/usr.local.lib64.conf && \
    ldconfig

# CMake
WORKDIR /root
RUN git clone https://github.com/Kitware/CMake.git CMake && \
    cd CMake && \
    git checkout tags/v3.15.0 && \
    ./bootstrap --system-curl --parallel=8 && \
    make -j 10 && \
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

ENV CC /usr/local/bin/gcc
ENV CXX /usr/local/bin/g++
RUN ldconfig

FROM build-stack as build

WORKDIR /root/test20
RUN echo "int main() { return 0; }" >> test.cpp
RUN g++ -o test -std=c++2a test.cpp
RUN /usr/local/bin/g++ --version

# LibTorch20
WORKDIR /root/libtorch20
ADD . .
RUN mkdir build && \
    cd build && \
    export CC=/usr/local/bin/gcc && \
    export CXX=/usr/local/bin/g++ && \
    cmake .. && \
    cmake --build . && \
    cmake --install .
