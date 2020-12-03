#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: loss_utils.py 
@time: 2019/09/23
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""


import numpy as np
import torch
import torch.nn.functional as F

NUM = 1.2#2.0
W = 1.0#10.0


def cal_loss_raw(pred, gold):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    #one_hot = F.one_hot(gold, pred.shape[1]).float()

    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss_raw = -(one_hot * log_prb).sum(dim=1)


    loss = loss_raw.mean()

    return loss,loss_raw

def mat_loss(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss



def cls_loss(pred, pred_aug, gold, pc_tran, aug_tran, pc_feat, aug_feat, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    cls_pc, _ = cal_loss_raw(pred, gold)
    cls_aug, _ = cal_loss_raw(pred_aug, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)

    feat_diff = 10.0*mse_fn(pc_feat,aug_feat)
    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(1.0-cls_pc_raw)**2).cuda()
    cls_diff = (torch.abs(cls_pc_raw - cls_aug_raw) * (parameters*2)).mean()
    cls_loss = cls_pc + cls_aug  + feat_diff# + cls_diff

    return cls_loss

def aug_loss(pred, pred_aug, gold, pc_tran, aug_tran, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold)
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)
    pc_con = F.softmax(pred, dim=-1)#.max(dim=1)[0]
    one_hot = F.one_hot(gold, pred.shape[1]).float()
    pc_con = (pc_con*one_hot).max(dim=1)[0]

     
    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(pc_con) * NUM).cuda()
    
    # both losses are usable
    aug_diff = W * torch.abs(1.0 - torch.exp(cls_aug_raw - cls_pc_raw * parameters)).mean()
    #aug_diff =  W*torch.abs(cls_aug_raw - cls_pc_raw*parameters).mean()
    aug_loss = cls_aug + aug_diff

    return aug_loss




