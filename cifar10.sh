#!/bin/bash
OUTPUT_DIR1=$1
DATA_DIR=$2

python train.py \
    --config configs/train/cifar10_ET.yml \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR1}

python test.py \
    --config configs/test/cifar10.yml \
    --checkpoint ${OUTPUT_DIR1}/best.ckpt \
    --data_dir ${DATA_DIR} \
    --csv_path ${OUTPUT_DIR1}/results.csv
