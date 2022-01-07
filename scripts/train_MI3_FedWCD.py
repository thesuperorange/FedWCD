import os
#FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
FILEPATH='/home/superorange5/MI3_dataset/MI3_dataset/JPEGImages/'
NET='vgg16'
WORKING_FOLDER='/home/superorange5/Research/FedWCD/'
scene_list = ['Doorway','Pathway']
#model_name_list = ['faster_rcnn_1_20_529.pth','faster_rcnn_1_20_521.pth','faster_rcnn_1_20_453.pth','faster_rcnn_1_20_265.pth'] #'faster_rcnn_1_20_467.pth',
DATASET='MI3'
mode='FedWCD' #'FedAvg'

test_data = 'all'


for i,scene in enumerate(scene_list):
    
    model_sub_dir=mode+'_no'+scene+'_'+test_data
    target_scene = 'MI3_train_'+scene
    

    if mode=='FedWCD':
        if test_data=='all':
            testdata_pkl = WORKING_FOLDER+'data/pickle/MI3_test_3325.pkl'
        else:
            testdata_pkl = WORKING_FOLDER+'data/pickle/MI3_test_'+scene+'.pkl'
        command = 'python {}trainval_net_wkAvg.py --dataset {} --net {} --bs 24 --nw 2 --lr 0.01 --cuda --epochs 3 --round 10 --save_sub_dir {} --target_scene {} --wk --testdata_pkl {} --mGPU'.format(WORKING_FOLDER,DATASET, NET,model_sub_dir,target_scene,testdata_pkl)
    elif mode=='FedAvg':
        command = 'python {}trainval_net_wkAvg.py --dataset {} --net {} --bs 24 --nw 2 --lr 0.01 --cuda --epochs 3 --round 10 --save_sub_dir {} --target_scene {} --mGPU'.format(WORKING_FOLDER,DATASET, NET,model_sub_dir,target_scene)

    
    print(command)
    os.system(command)

    
        
        


