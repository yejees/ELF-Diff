#!/bin/bash

# 1. environment setting
conda create -n ELF python=3.9
conda activate ELF
conda install pytorch torchvision torchaudio pytorch-cuda=11.4 -c pytorch -c nvidia
pip install -r requirements.txt