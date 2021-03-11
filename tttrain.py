import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo4 import YoloBody
from nets.yolo_training import Generator, YOLOLoss
from utils.dataloader import YoloDataset, yolo_dataset_collate


# 获得所有的类
# classes_path为在model_data文件夹下的yolo_classes.txt
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

# 获得所有的anchor
# anchors_path为model_data文件夹下的yolo_anchors.txt
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return anchors
    # return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

# 获得模型params里的学习率(lr)
# optimizer使用的是optim.Adam()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 一个eopch的训练过程
'''
net: 模型
yolo_losses: 长度为3的loss损失函数列表
eopch: 当前为第几个epoch，第1阶段为0-50，第2阶段为50-100
gen: 训练集DataLoader
gen_val: 验证集DataLoader
'''
def fit_one_epoch(net, yolo_losses, epoch, gen, gen_val):
    # 训练总loss
    train_total_loss = 0
    # 验证总loss
    val_total_loss = 0

    print('开始训练……')
    for iteration, batch in enumerate(gen):
        # iter大于等于batch总数，意味着1个epoch结束
        if iteration >= epoch_size:
            break
        # images: (bs, 3, 416, 416)，numpy
        # targets: bs * (n, 5)，numpy，n为1张图片上box的数量
        images, targets = batch[0], batch[1]
        # images和targets是输入数据，没必要对他们进行求导
        # 不使用torch.no_grad()也可以，使用的话可以加快GPU速度、减少显存占用
        # with torch.no_grad():
        images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
        targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        train_total_loss += loss
    print('训练结束，当前学习率为%f。' % get_lr(optimizer))

    # 将模型调回eval模式
    net.eval()
    print('开始验证……')
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_size_val:
            break
        images_val, targets_val = batch[0], batch[1]

        with torch.no_grad():
            images_val = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
            targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
            
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_total_loss += loss
    print('验证结束。')
    
    print('Epoch: '+ str(epoch + 1) + '/' + str(End_Epoch))
    print('训练总损失: %.4f || 验证总损失: %.4f ' %
          (train_total_loss / (epoch_size + 1),
           val_total_loss / (epoch_size_val + 1))
          )
    
    print('保存第%d轮模型权重。' % (epoch + 1))
    torch.save(model.state_dict(), 
               'logs/Epoch%d-Train_loss%.4f-Val_Loss%.4f.pth' % 
               ((epoch + 1), 
                train_total_loss / (epoch_size + 1),
                val_total_loss / (epoch_size_val + 1))
               )
    # 将模型调回train模式
    net.train()


if __name__ == "__main__":

    #-----------------------------------#
    #   基础参数
    #-----------------------------------#
    # 输入图片的尺寸
    input_shape = (416, 416)
    # learning rate是否使用余弦退火衰减
    Cosine_lr = False
    # 使用mosaic数据增强
    mosaic = True
    # 使用GPU训练
    Cuda = True
    # 平滑标签
    smoooth_label = 0
    # 使用DataLoader装载数据
    Use_Data_Loader = True
    # 训练数据信息路径（路径 + x,y,w,h(真实像素坐标) + class）
    annotation_path = 'trainval.txt'
    # anchors信息路径
    anchors_path = 'model_data/yolo_anchors.txt'
    # classes信息路径
    classes_path = 'model_data/voc_classes.txt'
    # 预训练权重文件路径
    model_path = "model_data/yolo4_voc_weights.pth"
    
    
    #-----------------------------------#
    #   获得先验框和类、划分训练/验证集
    #-----------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    # 0.2用于验证，0.8用于训练
    val_split = 0.2
    with open(annotation_path) as f:
        lines = f.readlines()
    # 打乱数据顺序
    np.random.shuffle(lines)
    # 训练/验证集总数
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    
    
    #-----------------------------------#
    #   构建dataset
    #-----------------------------------#
    train_dataset = YoloDataset(lines[:num_train], input_shape, mosaic = mosaic)
    '''
    it = iter(train_dataset)
    for i in range(24):
        ii = next(it)
        a, b = ii[0], ii[1]
        print(i)
        print(a.shape, b.shape)
    '''
    val_dataset = YoloDataset(lines[num_train:], input_shape, mosaic = False)
    
    
    #-----------------------------------#
    #   创建模型、加载预训练权重
    #-----------------------------------#
    model = YoloBody(3, num_classes)
    # 加快模型训练的效率
    print('加载预训练权重……')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('加载完成！')
    
    
    #-----------------------------------#
    #   设置网络训练信息
    #-----------------------------------#
    net = model.train()
    if Cuda:
        # 设置多GPU训练
        net = torch.nn.DataParallel(model)
        # 在开始训练时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法
        cudnn.benchmark = True
        net = net.cuda()
    
    
    #-----------------------------------#
    #   建立loss函数
    #-----------------------------------#
    yolo_losses = []
    # 3因为有13、26、52三种尺寸的输出
    for i in range(3):
        yolo_losses.append(
            YOLOLoss(
                # 把anchors调整为[w, h]的格式，倒序是因为13、26、52的feature map分别对应大、中、小的anchor
                np.reshape(anchors, [-1, 2])[::-1, :], 
                num_classes,
                input_shape, 
                smoooth_label, 
                Cuda)
            )
    
    
    #-----------------------------------#
    #   第一阶段训练
    #-----------------------------------#
    # 初始学习率
    lr = 1e-3
    Batch_size = 4
    Init_Epoch = 0
    End_Epoch = 50
    
    # 设置优化器
    optimizer = optim.Adam(net.parameters(), lr, weight_decay = 5e-4)
    # 设置学习率衰减方式
    if Cosine_lr:
        # CosineAnnealingLr: 让lr随着epoch的变化图类似于cos
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5, eta_min = 1e-5)
    else:
        # StepLR: 每过step_size个epoch，做一次更新
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.95)
    # 装载数据
    gen = DataLoader(train_dataset, 
                     shuffle = True, 
                     batch_size = Batch_size, 
                     num_workers = 4, 
                     pin_memory = True,
                     drop_last = True, 
                     collate_fn = yolo_dataset_collate)
    '''
    it = iter(gen)
    for i in range(6):
        ii = next(it)
        a, b = ii[0], ii[1]
        print(i)
        print(a.shape, b[0].shape, b[1].shape, b[2].shape, b[3].shape)
    '''
    gen_val = DataLoader(val_dataset, 
                         shuffle = True, 
                         batch_size = Batch_size, 
                         num_workers = 4, 
                         pin_memory = True, 
                         drop_last = True, 
                         collate_fn = yolo_dataset_collate)
    
    # 训练阶段1个epoch含有多少个batch
    epoch_size = max(1, num_train // Batch_size)
    # 验证阶段1个epoch含有多少个batch
    epoch_size_val = max(1, num_val // Batch_size)
    # 冻结一定部分训练，主干特征提取网络的特征是通用的，冻结起来可以加快训练效率
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    for epoch in range(Init_Epoch, End_Epoch):
        fit_one_epoch(net, yolo_losses, epoch, gen, gen_val)
        lr_scheduler.step()
    
    
    #-----------------------------------#
    #   第二阶段训练
    #-----------------------------------#
    lr = 1e-4
    Batch_size = 2
    Init_Epoch = 50
    End_Epoch = 100
    
    optimizer = optim.Adam(net.parameters(), lr, weight_decay = 5e-4)
    if Cosine_lr:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5, eta_min = 1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.95)
    
    gen = DataLoader(train_dataset,
                     shuffle = True, 
                     batch_size = Batch_size, 
                     num_workers = 4, 
                     pin_memory = True,
                     drop_last = True, 
                     collate_fn = yolo_dataset_collate)
    
    gen_val = DataLoader(val_dataset, 
                         shuffle = True, 
                         batch_size = Batch_size, 
                         num_workers = 4, 
                         pin_memory = True, 
                         drop_last = True, 
                         collate_fn = yolo_dataset_collate)
    
    epoch_size = max(1, num_train // Batch_size)
    epoch_size_val = max(1, num_val // Batch_size)
    # 解冻后训练，对主干特征提取网络的参数进行微调
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    for epoch in range(Init_Epoch,End_Epoch):
        fit_one_epoch(net, yolo_losses, epoch, gen, gen_val)
        lr_scheduler.step()
