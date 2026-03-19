# Sound Recognition with Deep Learning

A deep learning project for environmental sound classification using the ESC-50 dataset.

## Overview

Trains CNN models to classify audio clips into 50 sound categories. Audio files are converted to mel spectrograms and fed into a convolutional neural network.

## Dataset

[ESC-50](https://github.com/karolpiczak/ESC-50) — 2000 labeled audio clips across 50 environmental sound classes, split into 5 folds.

## Models

- **SimpleCNN** — custom 4-layer CNN
- **ResNetAudio** — pretrained ResNet-18 adapted for single-channel spectrogram input


Trains SimpleCNN for 40 epochs and saves the model as `SimpleCNN.pth`.
