#!/bin/bash

#### Run from command line (no Caliban)

OPT="--arch=hhnmlpb --dataset=FashionMNIST --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output"

CUDA_VISIBLE_DEVICES=0 python rotation_hhn.py $OPT --nlayers=1 --width=32 --dimensions=8
