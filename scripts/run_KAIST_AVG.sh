#!/bin/bash
#FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
FILEPATH='/home/superorange5/MI3_dataset/MI3_dataset/JPEGImages/'
NET='vgg16'
model_sub_dir='wkAVG_noStaircase'
DATASET='MI3'
for round in $(seq 1 10)
do
    
        echo round=$round
        echo python demoKAIST2.py  --net ${NET} --model_name faster_rcnn_${DATASET}_AVG_${round}.pth --cuda --load_dir models --image_dir $FILEPATH --output_folder ${DATASET}_fasterRCNN_${NET}-AVG_${round}
        python demoKAIST2.py  --dataset ${DATASET} --net ${NET} --model_name faster_rcnn_${DATASET}_AVG_${round}.pth --cuda --load_dir models --model_sub_dir $model_sub_dir --image_dir $FILEPATH --output_folder ${model_sub_dir}/${DATASET}_fasterRCNN_${NET}-AVG_${round}
        
done

