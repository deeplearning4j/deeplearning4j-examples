# Keras with Deeplearning4j

**Currently in ALPHA**

This directory contains code for running Keras with a Deeplearning4j backend. The example requires
Docker to run. Keras is used as a Python API for Deeplearning4j, and you can import, train, evaluate, predict,
and save your model. Both Functional and Sequential models are supported.

## Using

![Docker resource management](https://raw.githubusercontent.com/crockpotveggies/dl4j-examples/keras-examples/dl4j-keras-examples/src/main/resources/jupyter-home.jpg)

Jupyter is the simplest way to use this example. Build and run the included Dockerfile (you may also download
a pre-built one from Deeplearning4j's BinTray repository, instructions below). Once the container is running, copy
the token printed to the console and open the URL given (you may need to map your Docker port).

Inside Jupyter there will be several examples you can run. Copy the code into a Python Notebook and execute.

Note the `dl4j.install_dl4j_backend()`. You'll need to pass your model reference to this method each time you
  create a new instance.

## Pre-built Docker

```sh
    $ docker pull skymindio-docker-dl4j-examples.bintray.io/keras-dl4j:latest
    $ docker run -it skymindio-docker-dl4j-examples.bintray.io/keras-dl4j:latest
```

## Running

Once you've pulled your image, you will need to grab the Jupyter token from the console. Note that you will
 need the console for Deeplearning4j output when `model.fit()` and other Keras operations are running.

For convenience, the Kitematic interface allows you to view console output, open Jupyter in your web browser on the
 correct web port, and access an interactive shell.

![Docker resource management](https://raw.githubusercontent.com/crockpotveggies/dl4j-examples/keras-examples/dl4j-keras-examples/src/main/resources/kitematic-provisioning.jpg)

## System Resources

If you're using the Docker GUI installed on your machine, you may need to increase the memory and CPU available to
your container. We suggest using at least 8GB of RAM and 2 CPUs.

![Docker resource management](https://raw.githubusercontent.com/crockpotveggies/dl4j-examples/keras-examples/dl4j-keras-examples/src/main/resources/docker-provisioning.jpg)

Otherwise, consult the [Docker documentation](https://docs.docker.com) for increasing available resources for your container.

## Local Dockerfile

Build the image:
```sh
    $ docker build  -t keras-dl4j .
```

Bring it up:
```sh
    $ docker run -p 8888:8888 -it keras-dl4j
```

or use `docker-compose`
```sh
    $ docker-compose up
```
