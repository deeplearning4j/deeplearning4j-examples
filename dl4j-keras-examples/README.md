# Keras with Deeplearning4j

**Currently in ALPHA**

This directory contains code for running Keras v1.0.1 with a Deeplearning4j backend. The example requires
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
```

## Step-by-step

* Install [Docker Toolbox](https://www.docker.com/products/docker-toolbox)

During the install process, you will be offered the choice between the Docker Quickstart Terminal and Kitematic. Choose Kitematic, which will have you sign up for Docker Hub.

* Git clone the dl4j-examples branch below with the following line:


        git clone https://github.com/crockpotveggies/dl4j-examples --branch keras-examples --single-branch

The main Kitematic dashboard will look like this.

![kitematic dashboard](https://deeplearning4j.org/img/kitematic-dashboard.png)

Click on "My Images" on the upper right. One of your images then should be `keras-dl4j`.

![kitematic my images](https://deeplearning4j.org/img/kitematic-my-images.png)

Click "Create".

![kitematic provisioning](https://deeplearning4j.org/img/kitematic-provisioning.jpg)

Click on the icon on the upper right that will open a browser window for Jupyter using localhost. You'll get a screen that requires a Jupyter token, which you need to copy and paste from the bash console with Kitematic.

![jupyter token](https://deeplearning4j.org/img/jupyter-token.png)

Once you paste your token in the field, you should see this.

![jupyter home](https://deeplearning4j.org/img/jupyter-home.jpg)

Click on "New" on the upper right and select Python 2 under notebooks, which will open up a new notebook in a new tab.

Click on the Reuters MLP example at the bottom of the file list. Copy the Keras code and paste it into the new Python 2 notebook. In that notebook, on the toolbar, select the "Run Cell" button with the arrow pointing right.

![python notebook](https://deeplearning4j.org/img/python-notebook.png)

That's going to start printing out results and logs at the bottom of the notebook, which will look like this.

![python notebook results](https://deeplearning4j.org/img/python-notebook-results.png)

While the results may say "loading Theano", we are actually hijacking Keras methods to make them run on the JVM with Py4J.

## System Resources

If you're using the Docker GUI installed on your machine, you may need to increase the memory and CPU available to
your container. Go to the menu bar of your laptop screen, click on the Docker whale icon, and select Preferences/Advanced. There, you can increase the amount of memory allocated to Docker. We suggest 8GB if you can spare them. If you do adjust the amount of memory allocated to Docker, you will need to restart it by clicking on the button on the lower right.

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
