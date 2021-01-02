#!/bin/bash

IMAGE=
PREFIX=
MODEL="-model demo"
DEBUG=

if [ $# -ge 1 ]; then
#   IMAGE="-image $(printf '%q' '$1')"
    IMAGE="-image $1"
fi

if [ $# -ge 2 ]; then
    PREFIX="-prefix $2"
fi

if [ $# -ge 3 ]; then
    MODEL="-model eu"
    DEBUG="-debug $3"
fi

/usr/bin/python3 src/main.py $IMAGE $PREFIX $DEBUG $MODEL

