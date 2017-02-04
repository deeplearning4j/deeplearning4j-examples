# This image creates a basic setup for Keras with DL4J as a backend
#
# Based on the https://deeplearning4j.org/buildinglocally guide

FROM maven:alpine
MAINTAINER pkoperek@gmail.com

RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/community >> /etc/apk/repositories
RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/testing >> /etc/apk/repositories
RUN apk add --update git cmake gcc make g++ py-pip python-dev openblas openblas-dev hdf5 hdf5-dev

# Upgrade pip
RUN pip install --upgrade pip

# jupyter
RUN pip install jupyter

# Hack: http://serverfault.com/questions/771211/docker-alpine-and-matplotlib
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

# Install keras and other deps
RUN pip install keras
RUN pip install h5py py4j xxhash

# deeplearning4j
# RUN git clone --depth 1 https://github.com/deeplearning4j/deeplearning4j.git
RUN git clone --depth 1 -b keras-dl4j-api https://github.com/crockpotveggies/deeplearning4j.git; echo "20170202-2"

# main server
RUN git clone --depth 1 https://github.com/crockpotveggies/dl4j-examples.git
RUN cd dl4j-examples/deeplearning4j-keras-examples && mvn --settings /usr/share/maven/ref/settings-docker.xml clean install -DskipTests -Dmaven.javadoc.skip=true -p 'deeplearning4j-keras-examples'  && cd ../../

#
# keras-dl4j stuff
#

# Install keras-dl4j
RUN mkdir /keras-dl4j /root/.keras
COPY ./keras.json /root/.keras
ADD . /keras-dl4j
RUN cp deeplearning4j/python/. /keras-dl4j
RUN cd keras-dl4j && python setup.py sdist && pip install dist/kerasdl4j*tar.gz && cd ..

EXPOSE 8888
ENV KERAS_BACKEND theano

# adding `sh -c` solves the problem of dying kernels:
# https://github.com/ipython/ipython/issues/7062
CMD ["sh", "keras-dl4j/jupyter_start.sh"]
