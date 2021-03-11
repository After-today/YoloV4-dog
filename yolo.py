#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.yolo4 import YoloBody
from utils.utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": 'logs/Epoch100-Train_loss6.1517-Val_Loss4.5064.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_class.txt',
        "model_image_size" : (416, 416, 3),
        "confidence": 0.5,
        "iou" : 0.3,
        "cuda": True
    }

    # classmethod 修饰符对应的函数不需要实例化即可调用
    # 使用特殊参数cls而非self
    @classmethod
    def get_defaults(cls, attribute_name):
        if attribute_name in cls._defaults:
            return cls._defaults[attribute_name]
        else:
            return '没有定义的属性：' + attribute_name

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        # 类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        # 放到GPU上跑
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载预训练模型参数
        state_dict = torch.load(self.model_path, map_location = device)
        self.net.load_state_dict(state_dict)
        
        if self.cuda:
            # 指定使用哪几块GPU  '0, 1, 2'
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            # 等我有多张卡的时候再说  /(ㄒoㄒ)/~~
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
        print('Finished!')

        # 实例化三类size的anchor的DecodeBox并存入列表
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], 
                                               len(self.class_names), 
                                               (self.model_image_size[1], self.model_image_size[0])
                                               ))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), 
                self.colors)
            )

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        # 更改格式为float32
        photo = np.array(crop_img, dtype = np.float32)
        # 像素值压缩到0-1之间
        photo /= 255.0
        # 读进来得图片是H*W*C，输入网络时要求C*H*W，因此在这里变换通道
        photo = np.transpose(photo, (2, 0, 1))
        # photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            # 把numpy格式的像素数组转为tensor
            images = torch.from_numpy(images)
            if self.cuda:
                # 把数据放到GPU上
                images = images.cuda()
            # YoloBody得到预测结果
            # self.net == self.net.forword(images)
            outputs = self.net(images)
            

        output_list = []
        for i in range(3):
            # 用第i个DecodeBox来处理第i个output
            output_list.append(self.yolo_decodes[i](outputs[i]))
        
        # 将13、26、52的output拼接到一起  bs * 10647 * [4+1+num_classes]
        output = torch.cat(output_list, 1)

        # 使用非极大似然抑制剔除一定区域内的重复框
        # bs * n * [(x1,y1,x2,y2)+obj_conf+class_conf+class_pred]
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres = self.confidence,
                                               nms_thres = self.iou)
        
        # 整理检测结果
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image
        
        # 根据score再筛选一遍，但是在non_max_suppression已经使用score筛选过了为什么还要筛选呢？
        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        # 根据筛选结果得到符合要求的score、label、bboxes
        top_score = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        # 将(x1,y1,x2,y2)分别扩展至n*1维，n为box总数
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条，得到原图上预测框(y1,x1,y2,x2)坐标(top,left,bottom,right)
        boxes = yolo_correct_boxes(top_ymin, 
                                   top_xmin, 
                                   top_ymax, 
                                   top_xmax, 
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), 
                                   image_shape)

        # 绘制检测结果
        font = ImageFont.truetype(font = 'model_data/simhei.ttf', size = np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        # 矩形框四边线条厚度
        # thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]
        thickness = int(max(np.ceil(np.shape(image)[0] / self.model_image_size[0]), 
                            np.ceil(np.shape(image)[1] / self.model_image_size[0]))) + 1
        
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_score[i]
            
            top, left, bottom, right = boxes[i]
            # top = top - 5
            # left = left - 5
            # bottom = bottom + 5
            # right = right + 5

            # top = max(0, np.floor(top + 0.5).astype('int32'))
            # left = max(0, np.floor(left + 0.5).astype('int32'))
            # bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            # right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            top = max(0, np.ceil(top).astype('int32'))
            left = max(0, np.ceil(left).astype('int32'))
            bottom = min(np.shape(image)[0], np.ceil(bottom).astype('int32'))
            right = min(np.shape(image)[1], np.ceil(right).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            # 返回使用指定字体对象显示给定字符串所需要的图像尺寸
            label_size = draw.textsize(label, font)
            # label = label.encode('utf-8')
            # print(label)
            
            # 如果顶部有文本框的空间，文本框放置在预测框左上方的外部
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            # 顶部没有文本框的空间，文本框放置在预测框左上方的内部
            else:
                # text_origin = np.array([left, top + 1])
                text_origin = np.array([left + 1, top + 1])

            # 绘制预测框的空心矩形
            draw.rectangle(
                [left, top, right, bottom], 
                outline = self.colors[self.class_names.index(predicted_class)],
                width = thickness
                )
            # 绘制文本框的实心矩形
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill = self.colors[self.class_names.index(predicted_class)]
                )
            # 绘制文本框内的文字
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font = font)
            # fill = (0, 0, 0) 文字颜色纯黑
            draw.text(text_origin, label, fill = (0, 0, 0), font = font)

            del draw
        return image

