#!/bin/bash
# Train ten different models for cifar100 vgg19 with batch norm

NUM_RUNS=9

for RUN_ID in `seq 0 ${NUM_RUNS}`;
do
    OVERRIDE="{\"gpu_ids\":[0],\"exp_name\":\"adaptive_no_finetune_constrained-cifar100-vgg19_bn-run${RUN_ID}\"}"
    echo "$OVERRIDE"
    python3 main.py configs/cifar100/adaptive_constrained_vgg19_bn_no_fine_tuning.json \
        --override ${OVERRIDE}
done
