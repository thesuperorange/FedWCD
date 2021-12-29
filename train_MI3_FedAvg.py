import os
#FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
FILEPATH='/home/superorange5/MI3_dataset/MI3_dataset/JPEGImages/'
NET='vgg16'

scene_list = ['Room','Bus'] #'Staircase','Doorway',,'Pathway'
#model_name_list = ['faster_rcnn_1_20_529.pth','faster_rcnn_1_20_521.pth','faster_rcnn_1_20_453.pth','faster_rcnn_1_20_265.pth'] #'faster_rcnn_1_20_467.pth',
DATASET='MI3'

for i,scene in enumerate(scene_list):
    
    model_sub_dir='FedAvg_no'+scene
    target_scene = 'MI3_train_'+scene

    command = 'python trainval_net_wkAvg.py --dataset {} --net {} --bs 24 --nw 2 --lr 0.01 --cuda --epochs 3 --round 10 --save_sub_dir {} --target_scene {}'.format(DATASET, NET,model_sub_dir,target_scene)

    
    print(command)
    os.system(command)

    
        
        


