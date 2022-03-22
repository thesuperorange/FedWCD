import os
#FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
FILEPATH='/home/superorange5/MI3_dataset/MI3_dataset/JPEGImages/'
NET='vgg16'

scene_list = ['Room','Doorway','Bus','Pathway'] #'Staircase',
model_name_list = ['faster_rcnn_1_20_529.pth','faster_rcnn_1_20_521.pth','faster_rcnn_1_20_453.pth','faster_rcnn_1_20_265.pth'] #'faster_rcnn_1_20_467.pth',
DATASET='MI3'

for i,scene in enumerate(scene_list):
    
    model_sub_dir='baseline_no'+scene

    command = 'python demoKAIST2.py  --dataset {} --net {} --model_name {} --cuda --load_dir models --model_sub_dir {} --image_dir {} --output_folder {}'.format(DATASET, NET, model_name_list[i], model_sub_dir, FILEPATH ,model_sub_dir)
    print(command)
#    os.system(command)

    
        
        


