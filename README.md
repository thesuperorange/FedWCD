# FedWCD

## Usage

### train
  ```
  python trainval_net_wkAvg.py --dataset <dataset> --net <backbone> --bs <batch size> --nw <num_worker> --lr <learning rate> --cuda --epochs <local epochs> --round <round> --save_sub_dir <> --target_scene <target scene>'
  ```
  
  * Example

```
  python trainval_net_wkAvg.py --dataset MI3/KAIST --net vgg16 --bs 24 --nw 2 --lr 0.01 --cuda --epochs 3 --round 10 --save_sub_dir KAIST_vgg16_rd2c --target_scene campus'
```

### test
```
python demoKAIST2.py  --dataset KAIST --net vgg16 --model_name faster_rcnn_KAIST_train_rd_1_1081.pth --cuda --load_dir models --model_sub_dir KAIST_vgg16_rd2c --image_dir /home/superorange/DATA/KAIST/JPEGImages --output_folder KAIST_vgg16_rd2c
```

### evaluation

1. KAIST standard evaluation metrics (AP & MR)

  https://github.com/thesuperorange/task-conditioned
  
  ```
  python faster_eval.py
  ```

2. AP only

  https://github.com/thesuperorange/deepMI3/tree/master/model_evaluation

```
 python pascalvoc.py -g <GT folder> -d <detection folder>
```

