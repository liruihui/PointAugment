#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: config.py 
@time: 2019/09/17
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import argparse
import os

def str2bool(x):
    return x.lower() in ('true')

parser = argparse.ArgumentParser('PointNet')
parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
parser.add_argument('--epoch',  default=250, type=int, help='number of epoch in training')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--learning_rate_a', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--no_decay', type=str2bool, default=False)
parser.add_argument('--noise_dim',  default=1024, type=int, help='dimension of noise')

parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain Augment')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate of learning rate')

parser.add_argument('--model_name', default='pointnet', help='classification model')
parser.add_argument('--log_dir', default='log', help='log_dir')
parser.add_argument('--data_dir', default='ModelNet40_Folder')
parser.add_argument('--epoch_per_save', type=int, default=5)
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--y_rotated', type=str2bool, default=True)
parser.add_argument('--augment', type=str2bool, default=False)
parser.add_argument('--use_normal', type=str2bool, default=False)
parser.add_argument('--restore', action='store_true')

opts = parser.parse_args()
