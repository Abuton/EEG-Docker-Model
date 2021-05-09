FROM jupyter/scipy-notebook

COPY data/train.csv ./train.csv
COPY data/test.csv ./test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python train.py
