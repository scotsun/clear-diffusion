#!/bin/bash

for f in ./mnist/*.yaml; do
  echo "$f"
  python run.py --config "$f"
done
