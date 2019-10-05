#!/bin/bash
# Train ten different models for cifar100 vgg19 with batch norm

NUM_RUNS=9

for RUN_ID in `seq 0 ${NUM_RUNS}`;
do
    OVERRIDE="{\
	\"gpu_ids\":[0],\
        \"exp_name\":\"cifar100-vgg19_bn-minimal-run${RUN_ID}\", \
        \"model\":{ \
            \"name\": \"models.cifar.vgg.vgg_from_layers\", \
	    \"kwargs\": { \
	        \"layers\": [1, 1,\"M\", 1, 1, \"M\", 1, 1, 1, 1, \"M\", 1, 1, 1, 1, \"M\", 1, 1, 1, 1, \"M\"], \
	        \"batch_norm\": true, \
	        \"classifier_input_features\": 1, \
	        \"num_classes\": 100 \
            } \
        } \
    }"
    echo "$OVERRIDE"
    python3 main.py configs/cifar100/classification_vgg19_bn.json \
        --override "${OVERRIDE}"
done
