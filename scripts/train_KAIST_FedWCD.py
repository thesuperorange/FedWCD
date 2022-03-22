import os
FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
#FILEPATH='/home/superorange5/MI3_dataset/MI3_dataset/JPEGImages/'
NET='vgg16'
WORKING_FOLDER='/home/superorange5/Research/FedWCD/'
#scene_list = ['Doorway','Pathway']
scene_list = ['campus','road','downtown']
#model_name_list = ['faster_rcnn_1_20_529.pth','faster_rcnn_1_20_521.pth','faster_rcnn_1_20_453.pth','faster_rcnn_1_20_265.pth'] #'faster_rcnn_1_20_467.pth',
DATASET='KAIST'
mode='FedWCD' #'FedAvg'

test_data = '20210119'


for i,scene in enumerate(scene_list):
    
    model_sub_dir=mode+'_no'+scene+'_'+test_data
    
    target_scene = 'KAIST_'+scene   #'MI3_train_'+scene
    

    if mode=='FedWCD':
       
        testdata_pkl = '/home/superorange5/Research/2019_deepMI3/deepMI3/faster-RCNN/testimg2252.pkl'
            
        command = 'python {}trainval_net_wkAvg.py --dataset {} --net {} --bs 24 --nw 2 --lr 0.001 --lr_decay_step 4 --cuda --epochs 3 --round 10 --save_sub_dir {} --target_scene {} --wk --testdata_pkl {} --mGPU'.format(WORKING_FOLDER,DATASET, NET,model_sub_dir,target_scene,testdata_pkl)
    elif mode=='FedAvg':
        command = 'python {}trainval_net_wkAvg.py --dataset {} --net {} --bs 24 --nw 2 --lr 0.001 --lr_decay_step 4 --cuda --epochs 3 --round 10 --save_sub_dir {} --target_scene {} --mGPU'.format(WORKING_FOLDER,DATASET, NET,model_sub_dir,target_scene)

    
    print(command)
    os.system(command)

    
        
        


