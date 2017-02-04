## Description

Code for python part of keras - dl4j integration.

## Using

There are two ways to use this API.

* Hijack/extend existing keras model
* Use the low-level API manually

## Docker spells

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