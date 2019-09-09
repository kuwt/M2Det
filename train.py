from __future__ import print_function
import os
import warnings
warnings.filterwarnings('ignore')

import time
import torch
import shutil
import argparse
from m2det import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from data import detection_collate
from configs.CC import Config
from utils.core import *

parser = argparse.ArgumentParser(description='M2Det Training')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg16.py')
parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO or custom dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Training Program                       |\n'
           '----------------------------------------------------------------------',['yellow','bold'])

logger = set_logger(args.tensorboard)

 ##################### config #############################
global cfg
cfg = Config.fromfile(args.config)

 ##################### net #############################
net = build_net('train', 
                size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                config = cfg.model.m2det_config)
init_net(net, cfg, args.resume_net) # init the network with pretrained weights or resumed weights

 ##################### cuda #############################
if args.ngpu>1:
    net = torch.nn.DataParallel(net)
if cfg.train_cfg.cuda:
    net.cuda()
    cudnn.benchmark = True

 #################### optimizer, criterion, priorbox ###########
optimizer = set_optimizer(net, cfg)
criterion = set_criterion(cfg)
priorbox = PriorBox(anchors(cfg))

with torch.no_grad():
    priors = priorbox.forward()
    if cfg.train_cfg.cuda:
        priors = priors.cuda()

 #################### __main__ ###################################
if __name__ == '__main__':

    ########### switch to train mode ##################
    net.train()
    epoch = args.resume_epoch
    print_info('===> Loading Dataset...',['yellow','bold'])

    ################ get training dataset ################
    print("dataset = {}".format(args.dataset)) 
    if (args.dataset == "COCO" or args.dataset == "VOC"):
        dataset = get_dataloader(cfg, args.dataset, 'train_sets')
    else:
        dataset = get_dataloaderTrainOrTrainValCustomSet(cfg, 'train_sets')
    print("dataset len = {}".format(len(dataset)))

    ################ get dataset param ################
    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    max_iter = getattr(cfg.train_cfg.step_lr,args.dataset)[-1] * epoch_size
    stepvalues = [_*epoch_size for _ in getattr(cfg.train_cfg.step_lr, args.dataset)[:-1]]
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    
    print("start_iter, max_iter = {} , {}".format(start_iter,max_iter))
    print ("epoch size = {}".format(epoch_size))
    
    print_info('===> Training M2Det on ' + args.dataset, ['yellow','bold'])
    ############## train loop ##############
    for iteration in range(start_iter, max_iter):
        
        ########### Each epoch update ###########
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, 
                                                  cfg.train_cfg.per_batch_size * args.ngpu, 
                                                  shuffle=True, 
                                                  num_workers=cfg.train_cfg.num_workers, 
                                                  collate_fn=detection_collate))
            if epoch % cfg.model.save_eposhs == 0:
                save_checkpoint(net, cfg, final=False, datasetname = args.dataset, epoch=epoch)
            epoch += 1
        
       ######## Each step update ############
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, cfg.train_cfg.gamma, epoch, step_index, iteration, epoch_size, cfg)
        
        #######  Get data   ############ 
        # batch images and groundtruth.
        # batch number = per_batch_size x ngpu
        # a single groundtruth = [bb_norm_x, bb_norm_y, bb_norm_width, bb_norm_height, bb_class]  in the form of torch tensor
        #################
        images, targets = next(batch_iterator)
        #print("images type = ", type(images))
        #print("images size = ", images.size())
        #print("images = ",images)
        #print("targets type = ", type(targets))
        #print("targets = ", targets)

        if cfg.train_cfg.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        
        ########### forward ############
        out = net(images)
         
        ### loss , now the same as ssd loss###
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        write_logger({'loc_loss':loss_l.item(),
                      'conf_loss':loss_c.item(),
                      'loss':loss.item()},logger,iteration,status=args.tensorboard)

        ### backward ###
        loss.backward()
        optimizer.step()

         ### logging ###
        load_t1 = time.time()
        print_train_log(iteration, cfg.train_cfg.print_epochs,
                            [time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),load_t1-load_t0,lr])
    ### save_checkpoint after all training finish ###
    save_checkpoint(net, cfg, final=True, datasetname=args.dataset,epoch=-1)
