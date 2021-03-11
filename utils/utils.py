from __future__ import division

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.ops import nms


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        # input为四维：bs * [3*(4+1+num_classes)] * input_scale * input_scale
        # 一共多少张图片  batch_size
        batch_size = input.size(0)
        # input_scale * input_scale
        # input_height = input.size(2)
        # input_width = input.size(3)
        input_scale = input.size(2)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        # stride_h = self.img_size[1] / input_height
        # stride_w = self.img_size[0] / input_width
        stride_scale = self.img_size[0] / input_scale

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(anchor_width / stride_scale, anchor_height / stride_scale) for anchor_width, anchor_height in self.anchors]

        # 四维的input转为五维
        #    bs * [3*(4+1+num_classes)] * input_scale * input_scale 
        # -> bs * 3 * input_scale * input_scale * (4+1+num_classes)
        # 4 -> x, y, w, h    1 -> obj_confidence    num_classes -> classes_pred_score
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_scale, input_scale).permute(0, 1, 3, 4, 2).contiguous()

        # x, y, obj_confidence, classes_pred_score需要sigmoid
        # w, h不需要
        # xywh维度均为 bs * 3 * input_scale * input_scale

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 获得置信度，是否有物体  obj_confidence
        conf = torch.sigmoid(prediction[..., 4])
        # 种类预测得分  classes_pred_score
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 数据类型 如果使用GPU
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor


        # 生成网格 batch_size*3*input_scale*input_scale
        # 网格加上x,y的偏移量即可得到调整后的候选框的x,y
        # grid_x = torch.linspace(0, input_scale - 1, input_scale).repeat(input_scale, 1).repeat(
        #     batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        # grid_y = torch.linspace(0, input_scale - 1, input_scale).repeat(input_scale, 1).t().repeat(
        #     batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        g_x, g_y = torch.linspace(0, input_scale - 1, input_scale), torch.linspace(0, input_scale - 1, input_scale)
        g_x, g_y = torch.meshgrid(g_x, g_y)
        # meshgrid生成的网格坐标和像素坐标系是相反的，所以grid_x用y，grid_y用x
        grid_x = g_y.repeat(batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = g_x.repeat(batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        # 分别取出w和h，然后扩充至和xywh同维度 即batch_size*3*input_scale*input_scale
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_scale * input_scale).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_scale * input_scale).view(h.shape)
        
        # 计算调整后的先验框中心与宽高
        # 创建一个和前四项值x,y,w,h shape相同的tensor
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        # Location Prediction
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())

        # anchor_left = grid_x - anchor_w/2 
        # anchor_top = grid_y - anchor_h/2 

        # rect1 = plt.Rectangle([anchor_left[0,0,5,5],anchor_top[0,0,5,5]],anchor_w[0,0,5,5],anchor_h[0,0,5,5],color="r",fill=False)
        # rect2 = plt.Rectangle([anchor_left[0,1,5,5],anchor_top[0,1,5,5]],anchor_w[0,1,5,5],anchor_h[0,1,5,5],color="r",fill=False)
        # rect3 = plt.Rectangle([anchor_left[0,2,5,5],anchor_top[0,2,5,5]],anchor_w[0,2,5,5],anchor_h[0,2,5,5],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # ax = fig.add_subplot(122)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())
        # plt.scatter(pred_boxes[0,:,5,5,0].cpu(),pred_boxes[0,:,5,5,1].cpu(),c='r')

        # pre_left = pred_boxes[...,0] - pred_boxes[...,2]/2 
        # pre_top = pred_boxes[...,1] - pred_boxes[...,3]/2 

        # rect1 = plt.Rectangle([pre_left[0,0,5,5],pre_top[0,0,5,5]],pred_boxes[0,0,5,5,2],pred_boxes[0,0,5,5,3],color="r",fill=False)
        # rect2 = plt.Rectangle([pre_left[0,1,5,5],pre_top[0,1,5,5]],pred_boxes[0,1,5,5,2],pred_boxes[0,1,5,5,3],color="r",fill=False)
        # rect3 = plt.Rectangle([pre_left[0,2,5,5],pre_top[0,2,5,5]],pred_boxes[0,2,5,5,2],pred_boxes[0,2,5,5,3],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # plt.show()

        # 用于将输出调整为相对于416x416的大小，移动和形变后的xywh乘以stride_scale即可
        _scale = torch.Tensor([stride_scale, stride_scale] * 2).type(FloatTensor)
        # 最后的输出3维 ——> bs * [3*input_scale*input_scale] * [4+1+num_classes]
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw) // 2, (h - nh) // 2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)
    # 边缘添加的灰条的offset
    offset = (input_shape - new_shape) / 2.
    # # 用yx和hw是因为input_shape、new_shape、offset都是先高后宽
    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis = -1)
    box_hw = np.concatenate((bottom - top, right - left), axis = -1)
    # yx坐标减去灰条，hw长度不变即可
    box_yx = (box_yx - offset)

    # (y,x,h,w)还原为(y1,x1,y2,x2)
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    # 计算出坐标比例
    box_mins, box_maxes = box_mins / new_shape, box_maxes / new_shape
    # 将(y1,x1,y2,x2)组合起来
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis = -1)
    # 转化为原图上预测框的坐标
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes

# def bbox_iou(box1, box2, x1y1x2y2=True):
#     """
#         计算IOU
#     """
#     if not x1y1x2y2:
#         b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#         b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#         b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#         b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
#     else:
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

#     inter_rect_x1 = torch.max(b1_x1, b2_x1)
#     inter_rect_y1 = torch.max(b1_y1, b2_y1)
#     inter_rect_x2 = torch.min(b1_x2, b2_x2)
#     inter_rect_y2 = torch.min(b1_y2, b2_y2)

#     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
#                  torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
#     b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
#     b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

#     iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

#     return iou


def non_max_suppression(prediction, num_classes, conf_thres = 0.5, nms_thres = 0.4):
    # 求左上角和右下角
    # new() 一个新的和prediction.shape相同的tensor
    box_corner = prediction.new(prediction.shape)
    # 左上角 x1 = x - w/2
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # 左上角 x1 = y - h/2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # 右下角 x2 = x + w/2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # 右下角 y2 = y + h/2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # 用左上角和右下角坐标(x1,y1,x2,y2)替换原来的坐标(x,y,w,h)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # 多少个batch output长度为多少
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 获得类别得分最大的值及index
        # class_conf是每一box类别得分最大的分值 class_pred是类别得分最大的分值所处的列index
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim = True)

        # 利用box得分进行第一轮筛选
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        # 根据box得分筛选结果  取出符合box得分要求的数据
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # detections为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 获得通过box得分要求的预测框包含哪几种类别
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # 获得某一类初步筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            # 参数1：box左上右下坐标；参数2：box得分 = obj_conf * class_conf；参数3：iou_threshold
            # keep：NMS过滤后的bouding boxes索引
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            
            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data
            
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue
                
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue
                
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox
