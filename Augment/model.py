#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: model.py
@time: 2019/09/17
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description:
"""
import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from Augment.pointnet import PointNetCls
from Augment.augmentor import Augmentor
import numpy as np
from tensorboardX import SummaryWriter
import sklearn.metrics as metrics
import Common.data_utils as d_utils
import random
from Common import loss_utils
from Common.ModelNetDataLoader import ModelNetDataLoader

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


class Model:
    def __init__(self, opts):
        self.opts = opts
        self.backup()
        self.set_logger()

    def backup(self):
        if not self.opts.restore:
            source_folder = os.path.join(os.getcwd(),"Augment")
            common_folder = os.path.join(os.getcwd(), "Common")
            os.system("cp %s/config.py '%s/model_cls.py.backup'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/model.py '%s/model.py.backup'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/augmentor.py '%s/augmentor.py.backup'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/%s.py '%s/%s.py.backup'" % (source_folder,self.opts.model_name, self.opts.log_dir, self.opts.model_name))
            os.system("cp %s/loss_utils.py '%s/loss_utils.py.backup'" % (common_folder,self.opts.log_dir))
            os.system("cp %s/ModelNetDataLoader.py '%s/ModelNetDataLoader.py.backup'" % (common_folder,self.opts.log_dir))

    def set_logger(self):
        self.logger = logging.getLogger("CLS")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.opts.log_dir, "log_train.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def train(self):

        self.log_string('PARAMETER ...')
        self.log_string(self.opts)
        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments
        writer = SummaryWriter(logdir=self.opts.log_dir)
        '''DATA LOADING'''
        self.log_string('Load dataset ...')

        trainDataLoader = DataLoader(ModelNetDataLoader(self.opts, partition='train'),
                                  batch_size=self.opts.batch_size, shuffle=True, drop_last=False)
        testDataLoader = DataLoader(ModelNetDataLoader(self.opts,partition='test'),
                                 batch_size=self.opts.batch_size, shuffle=False,)

        self.log_string("The number of training data is: %d" % len(trainDataLoader.dataset))
        self.log_string("The number of test data is: %d" % len(testDataLoader.dataset))

        '''MODEL LOADING'''
        num_class = 40
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.dim = 3 if self.opts.use_normal else 0

        classifier = PointNetCls(num_class).cuda()
        augmentor = Augmentor().cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            classifier = nn.DataParallel(classifier)
            augmentor = nn.DataParallel(augmentor)

        if self.opts.restore:
            self.log_string('Use pretrain Augment...')

            checkpoint = torch.load(self.opts.log_dir)
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('No existing Augment, starting training from scratch...')
            start_epoch = 0


        optimizer_c = torch.optim.Adam(
            classifier.parameters(),
            lr=self.opts.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.decay_rate
        )

        optimizer_a = torch.optim.Adam(
            augmentor.parameters(),
            lr=self.opts.learning_rate_a,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.decay_rate
        )
        if self.opts.no_decay:
            scheduler_c = None
        else:
            scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=20, gamma=self.opts.lr_decay)

        #scheduler_a = torch.optim.lr_scheduler.StepLR(optimizer_a, step_size=20, gamma=self.opts.lr_decay)
        scheduler_a = None

        global_epoch = 0
        best_tst_accuracy = 0.0
        blue = lambda x: '\033[94m' + x + '\033[0m'
        ispn = True if self.opts.model_name=="pointnet" else False
        '''TRANING'''
        self.logger.info('Start training...')
        PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()  # initialize augmentation

        for epoch in range(start_epoch, self.opts.epoch):
            self.log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.opts.epoch))
            if scheduler_c is not None:
                scheduler_c.step(epoch)
            if scheduler_a is not None:
                scheduler_a.step(epoch)

            for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                points, target = data
                target = target[:, 0]
                points, target = points.cuda(), target.cuda().long()

                points = PointcloudScaleAndTranslate(points)
                points = points.transpose(2, 1).contiguous()

                noise = 0.02 * torch.randn(self.opts.batch_size, 1024).cuda()

                classifier = classifier.train()
                augmentor = augmentor.train()
                optimizer_a.zero_grad()
                aug_pc = augmentor(points, noise)

                pred_pc, pc_tran, pc_feat = classifier(points)
                pred_aug, aug_tran, aug_feat = classifier(aug_pc)
                augLoss  = loss_utils.aug_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, ispn=ispn)

                augLoss.backward(retain_graph=True)
                optimizer_a.step()


                optimizer_c.zero_grad()
                clsLoss = loss_utils.cls_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, pc_feat,
                                              aug_feat, ispn=ispn)
                clsLoss.backward(retain_graph=True)
                optimizer_c.step()


            train_acc = self.eval_one_epoch(classifier.eval(), trainDataLoader)
            test_acc = self.eval_one_epoch(classifier.eval(), testDataLoader)

            self.log_string('CLS Loss: %.2f'%clsLoss.data)
            self.log_string('AUG Loss: %.2f'%augLoss.data)

            self.log_string('Train Accuracy: %f' % train_acc)
            self.log_string('Test Accuracy: %f'%test_acc)
         
            writer.add_scalar("Train_Acc", train_acc, epoch)
            writer.add_scalar("Test_Acc", test_acc, epoch)


            if (test_acc >= best_tst_accuracy) and test_acc >= 0.895:# or (epoch % self.opts.epoch_per_save == 0):
                best_tst_accuracy = test_acc
                self.log_string('Save model...')
                self.save_checkpoint(
                    global_epoch + 1,
                    train_acc,
                    test_acc,
                    classifier,
                    optimizer_c,
                    str(self.opts.log_dir),
                    self.opts.model_name)

            global_epoch += 1
        self.log_string('Best Accuracy: %f' % best_tst_accuracy)
        self.log_string('End of training...')
        self.log_string(self.opts.log_dir)


    def eval_one_epoch(self, model, loader):
        mean_correct = []
        test_pred = []
        test_true = []

        for j, data in enumerate(loader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = model.eval()
            pred, _, _= classifier(points)
            pred_choice = pred.data.max(1)[1]

            test_true.append(target.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
       
        return test_acc

    def save_checkpoint(self, epoch, train_accuracy, test_accuracy, model, optimizer, path, modelnet='checkpoint'):
        savepath = path + '/%s-%f-%04d.pth' % (modelnet, test_accuracy, epoch)
        print(savepath)
        state = {
            'epoch': epoch,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            #'model_state_dict': model.module.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)

    def log_string(self, msg):
        print(msg)
        self.logger.info(msg)


