#!/bin/bash
NET='vgg16'
DATASET='MI3'
SCENES='Bus Staircase Doorway Pathway Room'
METHOD='FedAvg'
for SCENE in $SCENES; do
	model_sub_dir=$METHOD'_no'$SCENES

	for round in $(seq 1 9)
	do
		rm -rf ../output/${METHOD}_no${SCENE}/${DATASET}_fasterRCNN_${NET}-AVG_${round}/ 
	done 
done

