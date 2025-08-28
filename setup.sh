#!/bin/bash

# nettoyage préalable
./tools/cleanup.sh

python3 src/data/getRawData.py
python3 src/data/setFeatures.py
python3 src/models/findBestParams.py
python3 src/models/trainModel.py
python3 src/models/predict.py
