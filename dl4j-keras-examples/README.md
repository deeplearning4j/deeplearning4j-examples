# Keras with Deeplearning4j

This directory contains code for running Keras with a Deeplearning4j backend. The example requires
Docker to run. Keras is used as a Python API for Deeplearning4j, and you can import, train, evaluate, predict,
and save your model. Both Functional and Sequential models are supported.

## Using

Jupyter is the simplest way to play with this example. Build and run the included Dockerfile (you may also download
a pre-built one from Deeplearning4j's BinTray repository, instructions below). Once the container is running, copy
the token logged to the console and open the URL given (you may need to map your Docker port).

Inside Jupyter there will be several examples you can run. Copy the code into a Python Notebook and execute.

Note the `dl4j.install_dl4j_backend()`. You'll need to pass your model reference to this method each time you
  create a new instance.

## Pre-built Docker

*Coming soon*

```sh
    $ docker pull {subject}-docker-{repo}.bintray.io/[{namespace}/]{docker_repo}[:{version}]
    $ docker run -it {subject}-docker-{repo}.bintray.io/[{namespace}/]{docker_repo}[:{version}]
```

## Local Dockerfile

Build the image:
```sh
    $ docker build . -t keras-dl4j
```

Bring it up:
```sh
    $ docker run -p 8888:8888 -it keras-dl4j
```

or use `docker-compose`
```sh
    $ docker-compose up
```
