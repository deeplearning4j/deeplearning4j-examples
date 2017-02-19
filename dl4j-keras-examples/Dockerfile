# This image creates a basic setup for Keras with DL4J as a backend
#
# Based on the https://deeplearning4j.org/buildinglocally guide

FROM anapsix/alpine-java:8_jdk
MAINTAINER Justin Long @crockpotveggies

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


# Install maven
ENV MAVEN_VERSION="3.2.5" \
    M2_HOME=/usr/lib/mvn

RUN apk add --update wget && \
  cd /tmp && \
  wget "http://ftp.unicamp.br/pub/apache/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz" && \
  tar -zxvf "apache-maven-$MAVEN_VERSION-bin.tar.gz" && \
  mv "apache-maven-$MAVEN_VERSION" "$M2_HOME" && \
  ln -s "$M2_HOME/bin/mvn" /usr/bin/mvn && \
  apk del wget && \
  rm /tmp/* /var/cache/apk/*

#
# build DL4J stack
#

# libnd4j
RUN git clone --depth 1 https://github.com/deeplearning4j/libnd4j.git
RUN cd libnd4j && ./buildnativeoperations.sh && cd ..

ENV LIBND4J_HOME /libnd4j

# nd4j
RUN git clone --depth 1 https://github.com/deeplearning4j/nd4j.git
RUN cd nd4j && mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests' && cd ..

# datavec
RUN git clone --depth 1 https://github.com/deeplearning4j/datavec.git
RUN cd datavec && mvn clean install -DskipTests -Dmaven.javadoc.skip=true && cd ..

# test resources
RUN git clone --depth 1 https://github.com/deeplearning4j/dl4j-test-resources.git
RUN cd dl4j-test-resources && mvn clean install && cd ..

# deeplearning4j
# RUN git clone --depth 1 https://github.com/deeplearning4j/deeplearning4j.git
RUN git clone --depth 1 -b keras-dl4j-api https://github.com/crockpotveggies/deeplearning4j.git; echo "20170217-1"
RUN cd deeplearning4j && mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0,!:deeplearning4j-ui-parent,!:deeplearning4j-ui_2.11,!:dl4j-spark_2.11,!:dl4j-spark-nlp_2.11,!:dl4j-spark-parameterserver_2.11,!:deeplearning4j-scaleout-parallelwrapper-parameter-server,!:dl4j-streaming_2.11,!:deeplearning4j-ui-remote-iterationlisteners,!:deeplearning4j-nlp-uima' && cd ..

#
# keras-dl4j stuff
#

# Install keras-dl4j
RUN mkdir /keras-dl4j /root/.keras
COPY ./keras.json /root/.keras
ADD . /keras-dl4j
RUN cp -r /deeplearning4j/python/kerasdl4j /keras-dl4j/kerasdl4j
RUN cp -r /deeplearning4j/deeplearning4j-keras/src /keras-dl4j/
# compile server
RUN cd keras-dl4j && mvn package && cd ..
# install python code
RUN cd keras-dl4j && python setup.py sdist && pip install dist/kerasdl4j*tar.gz && cd ..

EXPOSE 8888
ENV KERAS_BACKEND theano

# adding `sh -c` solves the problem of dying kernels:
# https://github.com/ipython/ipython/issues/7062
CMD ["sh", "keras-dl4j/jupyter_start.sh"]
