import os
#FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
FILEPATH='/home/superorange5/MI3_dataset/MI3_dataset/JPEGImages/'
NET='vgg16'

scene_list = ['Doorway','Pathway'] #['Staircase','Room','Bus'] #'Doorway',,'Pathway'
#model_name_list = ['faster_rcnn_1_20_529.pth','faster_rcnn_1_20_521.pth','faster_rcnn_1_20_453.pth','faster_rcnn_1_20_265.pth'] #'faster_rcnn_1_20_467.pth',
DATASET='MI3'
ROUND_NUM = 10
mode='FedWCD' #'FedAvg'
test_data = 'all'
for i,scene in enumerate(scene_list):
    model_sub_dir=mode+'_no'+scene+'_'+test_data
    for r in range(1,ROUND_NUM+1):
        model_name = 'faster_rcnn_'+DATASET+'_AVG_'+str(r)+'.pth'
        output_folder = model_sub_dir+'/'+DATASET+'_fasterRCNN_'+NET+'-AVG_'+str(r)

        command = 'python demoKAIST2.py  --dataset {} --net {} --model_name {} --cuda --load_dir models --model_sub_dir {} --image_dir {} --output_folder {}'.format(DATASET, NET, model_name, model_sub_dir, FILEPATH ,output_folder)
        print(command)
        os.system(command)

    

                
        


