#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: train_PA.py 
@time: 2019/09/17
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""
import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime

from Augment.model import Model
from Augment.config import opts


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    opts.log_dir = os.path.join(opts.log_dir,opts.model_name+"_cls", current_time)
    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)

    print('checkpoints:', opts.log_dir)

    model = Model(opts)
    model.train()

