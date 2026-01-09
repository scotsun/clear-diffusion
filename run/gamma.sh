#!/bin/bash

for f in ./config/mnist/*.yaml; do
  echo "$f"
  python run/train_clear_mnist.py --config "$f"
done
