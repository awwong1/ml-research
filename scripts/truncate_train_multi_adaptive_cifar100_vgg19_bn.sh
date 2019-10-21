#!/bin/bash

for RUN_ID in `seq 9 -1 1`;
do
    OVERRIDE="{\"gpu_ids\":[1],\"exp_name\":\"truncate_train-adaptive_constrained-cifar100-vgg19_bn-run0-${RUN_ID}\",\"truncate_train_data_ratio\":0.${RUN_ID}}"
    echo "$OVERRIDE"
    python3 main.py configs/cifar100/adaptive_constrained_vgg19_bn.json \
        --override ${OVERRIDE}
done
