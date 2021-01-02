#!/bin/bash

#MODEL="-model demo"
MODEL="-model eu"
DIR="-dir test"
DEBUG=

if [ $# -ge 3 ]; then
    MODEL="-model eu"
    DEBUG="-debug $3"
fi

/usr/bin/python3 src/test.py $DEBUG $MODEL $DIR

