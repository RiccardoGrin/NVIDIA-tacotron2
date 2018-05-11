#!/bin/bash

source activate pytorch_source

python train.py --output_directory=outdir --log_directory=logdir --checkpoint_path="/home/ubuntu/NVIDIA-tacotron2/outdir/checkpoint_1000"