# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb, rank_roidb_ratio, filter_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from frcnn_helper import *
from scipy.special import softmax

import pickle

import FedUtils
import KD_utils

#imdb_name = 'KAIST_train_cr'  
data_cache_path = 'data/cache'
imdb_classes =  ('__background__',  # always index 0
                          'person',
                          'people','cyclist'
                         )
#parties = len(imdb_list)




def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--save_sub_dir', dest='save_sub_dir',
                        help='directory to save models', default="",
                        type=str)
        
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # freeze base layer
    parser.add_argument('--f', dest='freeze',
                    help='freeze base layer or not',
                    action='store_true')
    parser.add_argument('--input', dest='imdb_name',
                        help='input imdb_name for loading pkl file')
    
    parser.add_argument('--teacher', dest='teacher',
                        help='input teacher model file path')
    
    parser.add_argument('--KD_mode', dest='KD_mode',
                    help='set KD params, default/off/softonly/hintonly/mask') 

    
    # params for KDv2
    parser.add_argument('--ws', dest='warm_step',
                    help='warm up the training',
                    default=400, type=int)
    parser.add_argument('--ilw', dest='imitation_loss_weigth',
                    help='imitation loss weight',
                    default=0.01, type=float)
    
    # weighted FedAvg
#     parser.add_argument('--wk', dest='wkFedAvg',
#                         help='using within class as weighted to average model weight',
#                         action='store_true')

#     # resume trained model
#     parser.add_argument('--r', dest='resume',
#                         help='resume checkpoint or not',
#                         action='store_true')
#     parser.add_argument('--resume_model_name', dest='resume_model_name',
#                         help='resume model name')
    
#     parser.add_argument('--checkround', dest='checkround',
#                         help='checkround to load model',
#                         default=1, type=int)
#     parser.add_argument('--checkepoch', dest='checkepoch',
#                         help='checkepoch to load model',
#                         default=1, type=int)
#     parser.add_argument('--checkpoint', dest='checkpoint',
#                         help='checkpoint to load model',
#                        default=0, type=int)
#     parser.add_argument('--round', dest='round',
#                     help='total rounds',
#                     default=10, type=int)
    
#     parser.add_argument('--k', dest='k',
#                         help='k of cluster #',
#                         default=5, type=int)


    args = parser.parse_args()
    return args

def load_client_dataset(imdb_name):
    #dataloader_list = []
    #iter_epochs_list = []
    #for imdb_name in imdb_list:
    pkl_file = os.path.join(data_cache_path, imdb_name + '_gt_roidb.pkl')

    with open(pkl_file, 'rb') as f:
        roidb = pickle.load(f)

    roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    train_size = len(roidb)
    print(train_size)
    iters_per_epoch = int(train_size / args.batch_size)
    print('iters_per_epoch: ' + str(iters_per_epoch))
    #iter_epochs_list.append(iters_per_epoch)
    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)
    #dataloader_list.append(dataloader)
    return dataloader,iters_per_epoch


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

    
# def initial_network(args):
    
#       # initilize the network here.
#     if args.net == 'vgg16':
#         fasterRCNN = vgg16(imdb_classes, pretrained=True, class_agnostic=args.class_agnostic)
#     elif args.net == 'res101':
#         fasterRCNN = resnet(imdb_classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
#     elif args.net == 'res50':
#         fasterRCNN = resnet(imdb_classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
#     elif args.net == 'res152':
#         fasterRCNN = resnet(imdb_classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
#     else:
#         print("network is not defined")
#         pdb.set_trace()

#     fasterRCNN.create_architecture()

#     if args.cuda:
#         fasterRCNN.cuda()

#     if args.mGPUs:
#         fasterRCNN = nn.DataParallel(fasterRCNN)
        
#     return fasterRCNN

# def getOptimizer(fasterRCNN,args):
#     lr = args.lr
#     params = []
#     for key, value in dict(fasterRCNN.named_parameters()).items():
#         if value.requires_grad:
#             if 'bias' in key:
#                 params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
#                     'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
#             else:
#                 params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
                
#     if args.optimizer == "adam":
#         lr = lr * 0.1
#         optimizer = torch.optim.Adam(params)

#     elif args.optimizer == "sgd":
#         optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM) 
#     return optimizer


def train(args,dataloader,imdb_name,iters_per_epoch, fasterRCNN_s, fasterRCNN_t, optimizer, KD_params):     
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    
    lr = args.lr

    
    
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN_s.train()
        #fasterRCNN_t.train()
        loss_temp = 0
        loss_sup_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)

            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            fasterRCNN_s.zero_grad()

            
            # KD params
            u=KD_params['u']
            v=KD_params['v']
            lambda_var = KD_params['lambda_var']
            gamma = KD_params['gamma']

            
            # student net
            rois, rcn_Ps, rcn_Rs, rpn_Lhard, rpn_LsL1, \
            rcn_Lhard, rcn_LsL1, rois_label, rpn_Ps, rpn_Rs, fmap_s, rcn_gt, rpn_gt, mask_batch = fasterRCNN_s(im_data, im_info, gt_boxes, num_boxes)
            # teacher net
            _, rcn_Pt, rcn_Rt, rpn_Lhard_t, rpn_LsL1_t, rcn_Lhard_t, rcn_LsL1_t, _, rpn_Pt, rpn_Rt, fmap_t, rcn_gt_t,rpn_gt_t, _ = fasterRCNN_t(im_data, im_info, gt_boxes, num_boxes)
            
#             print('[bounded_regression_loss] **RPN** Rs v.s. gt: %.4f' % (F.smooth_l1_loss(rpn_Rs, rpn_gt).item()))
#             #print('[bounded_regression_loss] **RPN** Rt v.s. gt_s: %.4f' % (F.mse_loss(rpn_Rt, rpn_gt).item()))
#             print('[bounded_regression_loss] **RPN** Rt v.s. gt_t: %.4f' % (F.smooth_l1_loss(rpn_Rt, rpn_gt_t).item()))
            
#             print('[bounded_regression_loss] **RCN** Rs v.s. gt: %.4f' % (F.smooth_l1_loss(rcn_Rs, rcn_gt).item()))
#             #print('[bounded_regression_loss] **RCN** Rt v.s. gt_s: %.4f' % (F.mse_loss(rcn_Rt, rcn_gt).item()))
#             print('[bounded_regression_loss] **RCN** Rt v.s. gt_t: %.4f' % (F.smooth_l1_loss(rcn_Rt, rcn_gt_t).item()))

            
            #----------------------KD v2------------------
            
            mask_list = []
            for mask in mask_batch:
                mask = (mask > 0).float().unsqueeze(0)
                mask_list.append(mask)
            mask_batch = torch.stack(mask_list, dim=0)
            norms = mask_batch.sum() * 2
            
            #stu_feature_adap = fasterRCNN.stu_feature_adap(stu_feature)  # comment out because we use the same backbone for both t&s
#             sup_loss = (torch.pow(fmap_t - fmap_s, 2) * mask_batch).sum() / norms
#             sup_loss = sup_loss * args.imitation_loss_weigth

            

                            
            
#------------------------old KD v1           
            
            # Lhint=|V-Z|^2
            loss_hint = F.mse_loss(fmap_s, fmap_t)
            
            ### RPN part
            
            
            if step % args.disp_interval == 0:
                print("RPN")
            u=0.5
            v=0.5
            rpn_loss_cls, rpn_loss_reg = \
                KD_utils.KD_getloss(rpn_Ps, rpn_Pt, rpn_Rs, rpn_Rt, rpn_gt,rpn_gt_t, rpn_LsL1, rpn_Lhard,step,args.mGPUs, 1.0, 1.5, u, v)
            
             ### RCN part
            if step % args.disp_interval == 0:
                print("RCN")
            u=1
            v=0
            rcn_loss_cls, rcn_loss_reg = \
                KD_utils.KD_getloss(rcn_Ps, rcn_Pt, rcn_Rs, rcn_Rt, rcn_gt,rcn_gt_t, rcn_LsL1, rcn_Lhard,step, args.mGPUs,1.0, 1.5, u, v)

            
            if args.KD_mode == 'mask':
            
                loss = rpn_Lhard.mean() + rpn_LsL1.mean() \
                       + rcn_Lhard.mean() + rcn_LsL1.mean()
                #loss_temp += loss.data[0]
                loss_temp += loss.item()

                if (step < args.warm_step and epoch == 1) :#or args.turn_off_imitation:
                    sup_loss = sup_loss * 0
                loss += sup_loss
                loss_sup_temp += sup_loss            
            else:
            
                loss = gamma * loss_hint + rpn_loss_cls.mean() + rcn_loss_cls.mean() + lambda_var*( rpn_loss_reg.mean() + rcn_loss_reg.mean())
                loss_temp += loss.item()
            
            #      print('loss={}'.format(loss))
            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN_s, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                    loss_sup_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_Lhard = rpn_Lhard.mean().item()
                    loss_rpn_LsL1 = rpn_LsL1.mean().item()
                    loss_rcn_Lhard = rcn_Lhard.mean().item()
                    loss_rcn_LsL1 = rcn_LsL1.mean().item()
                    
                    loss_rcn_LsL1_t = rcn_LsL1_t.mean().item()
                    loss_rpn_LsL1_t = rpn_LsL1_t.mean().item()
                    
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_reg.mean().item()
                    loss_rcnn_cls = rcn_loss_cls.mean().item()
                    loss_rcnn_box = rcn_loss_reg.mean().item()
                    
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_Lhard = rpn_Lhard.item()
                    loss_rpn_LsL1 = rpn_LsL1.item()
                    loss_rcn_Lhard = rcn_Lhard.item()
                    loss_rcn_LsL1 = rcn_LsL1.item()
                    
                    loss_rcn_LsL1_t = rcn_LsL1_t.item()
                    loss_rpn_LsL1_t = rpn_LsL1_t.item()
                    
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_reg.item()
                    loss_rcnn_cls = rcn_loss_cls.item()
                    loss_rcnn_box = rcn_loss_reg.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    
                print("[t]rcn_LsL1_t: %.4f, [s]rcn_LsL1: %.4f, [t]rpn_LsL1_t: %.4f, [s]rpn_LsL1 %.4f" \
                      % (loss_rcn_LsL1_t, loss_rcnn_box, loss_rpn_LsL1_t, loss_rpn_LsL1))

                print("[epoch %2d][iter %4d/%4d] loss: %.4f, loss_sup: %.4f, lr: %.2e" \
                      % (epoch, step, iters_per_epoch, loss_temp, loss_sup_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                print("\t\t\trcn_Lhard: %.4f, rcn_LsL1: %.4f, rpn_Lhard: %.4f, rpn_LsL1 %.4f" \
                      % (loss_rcn_Lhard, loss_rcn_LsL1, loss_rpn_Lhard, loss_rpn_LsL1))
                
                print("\t\t\tloss_hint: %.4f" \
                      % (loss_hint.item()))
                

                loss_temp = 0
                start = time.time()
        #if epoch == args.max_epochs + 1 :
        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(imdb_name, epoch, step))
        save_checkpoint({
         
          'epoch': epoch + 1,
          'model': fasterRCNN_s.module.state_dict() if args.mGPUs else fasterRCNN_s.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
        
            
        
    return fasterRCNN_s    
        
# def avgWeight(model_list,ratio_list):
#     model_tmp=[None] * parties
#     #optims_tmp=[None] * parties

#     for idx, my_model in enumerate(model_list):
        
#         model_tmp[idx] = my_model.state_dict()


#     for key in model_tmp[0]:    
#         #print(key)
#         model_avg = 0

#         for idx, model_tmp_content in enumerate(model_tmp):     # add each model              
#             model_avg += ratio_list[idx] * model_tmp_content[key]
            
#         for i in range(len(model_tmp)):  #copy to each model            
#             model_tmp[i][key] = model_avg
#     for i in range(len(model_list)):    
#         model_list[i].load_state_dict(model_tmp[i])
        
#     return model_list  #, optims_tmp
                    
# def load_model(model_path, args):
#     model = initial_network(args)
#     checkpoint = torch.load(model_path)
    
#     if args.mGPUs:
#         model.module.load_state_dict(checkpoint['model'])
#     else:
#         model.load_state_dict(checkpoint['model'])
    
#     optimizer = getOptimizer(model,args)
#     optimizer.load_state_dict(checkpoint['optimizer'])
    
#     start_round = checkpoint['round']
#     return model,optimizer, start_round


    
if __name__ == '__main__':

    args = parse_args()
    
    imdb_name = args.imdb_name

    print('Called with args:')
    print(args)
    

    # ---- KD parameters setting ----
    # u: balance the hard and soft losses
    # v: ratio of bounded_reg
    # gamma: ratio of hint
    # lambda: ratio of reg
    
    KD_mode = args.KD_mode
    
    if KD_mode=='off':
        KD_params ={'u':1, 'v':0, 'gamma':0, 'lambda_var':1}  # lambda: don't care
    elif KD_mode=='softonly':
        KD_params ={'u':0, 'v':0, 'gamma':1, 'lambda_var':0}  # v: don't care because lambda already set to 0
    
    elif KD_mode == 'hintonly':
        KD_params ={'u':1, 'v':0, 'gamma':0.5, 'lambda_var':1}
    
    else:
        #default / mask (didn't use any of them)
        KD_params ={'u':0.5, 'v':0.5, 'gamma':0.5, 'lambda_var':1}
        #print("wrong KD_mode")
        #exit() 
 

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")



    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + args.save_sub_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    dataloader,iters_per_epoch  = load_client_dataset(imdb_name)
    #dataloader = dataloader_list[0]
    print('# worker' + str(args.num_workers))
    # initilize the tensor holder here.
    

    if args.cuda:
        cfg.CUDA = True
        
    teacher_model_path = args.teacher
    fasterRCNN_teacher = FedUtils.load_model_KD(imdb_classes, teacher_model_path, args, cfg)
    fasterRCNN_student = FedUtils.initial_network_KD(imdb_classes, args)
    
    

    
    # freeze all VGG layer
    if args.freeze == True:
        if args.mGPUs:
            for p in fasterRCNN_student.module.RCNN_base.parameters(): p.requires_grad=False
        else:
            for p in fasterRCNN_student.RCNN_base.parameters(): p.requires_grad=False
                
            
    optimizer = FedUtils.getOptimizer(fasterRCNN_student,args,cfg)
  
        
    train(args,dataloader,imdb_name,iters_per_epoch, fasterRCNN_student, fasterRCNN_teacher, optimizer, KD_params)
        


