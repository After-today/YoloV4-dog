import math
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from nets.yolo_training import Generator
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# from utils.utils import bbox_iou, merge_bboxes
from utils.utils import merge_bboxes


class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size, mosaic = True):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.mosaic = mosaic
        self.flag = True

    def __len__(self):
        return self.train_batches

    def rand(self, a = 0, b = 1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        # index 等于batch_size个数（如bs=4，则index有4个数）
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        if self.mosaic:
            # index+4<n是为了保证mosaic数据增强的时候需要index位置的图片和该位置往后的三张图片
            if self.flag and (index + 4) < n:
                img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.image_size[0:2])
            else:
                img, y = self.get_random_data(lines[index], self.image_size[0:2])
            # 避免每次都用mosaic增强
            self.flag = bool(1 - self.flag)
        else:
           img, y = self.get_random_data(lines[index], self.image_size[0:2])

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)
        return tmp_inp, tmp_targets
    
    #-----------------------------------#
    #   图片色域调整
    #-----------------------------------#
    def Color_gamut_transform(self, image, hue, sat, val):
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image

    #-----------------------------------#
    #   标注框后处理
    #-----------------------------------#
    def post_process_box(self, box, w, nw, iw, nh, ih, dx, dy, flip):
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            # 如果发生水平翻转，只需要改动x1,y1,x2,y2的x1,x2即可
            # x1 = iw - x2;     x2 = iw - x1
            if flip:
                box[:, [0, 2]] = iw - box[:, [2, 0]]
            # 处理box的x1,x2，根据原尺寸和缩放后的尺寸以及dx
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            # 处理box的y1,y2，根据原尺寸和缩放后的尺寸以及dy
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            # paste过程中可能会损失掉图片的部分区域，进而可能会损失掉box的部分区域
            # 因此需要对左上角和右小角判断是否超出416*416区域
            # 左上角坐标如果有小于0的更改为0
            box[:, 0:2][box[:, 0:2] < 0] = 0
            # 右下角坐标如果有超出图片尺寸的更改为416
            box[:, 2:4][box[:, 2:4] > w] = w
            # box[:, 2][box[:, 2] > w] = w
            # box[:, 3][box[:, 3] > h] = h

            # 宽 = x2 - x1
            box_w = box[:, 2] - box[:, 0]
            # 高 = y2 - y1
            box_h = box[:, 3] - box[:, 1]

            # logical_and/or/not(x1, x2)  x1,x2的逻辑与或非，宽高均大于1，保留有效框
            # 如极端情况，box小且在右小角，经过paste整个框都在416*416区域外
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            # 创建一个和box等尺寸的空array，避免=赋值问题
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        return box_data

    #-----------------------------------#
    #   不使用mosaic方法的数据增强
    #-----------------------------------#
    def get_random_data(self, annotation_line, input_shape, jitter = .3, hue = .1, sat = 1.5, val = 1.5):
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', 
                              (w, h),
                              (np.random.randint(0, 255), 
                               np.random.randint(0, 255), 
                               np.random.randint(0, 255))
                              )
        new_image.paste(image, (dx, dy))
        image = new_image

        # 是否翻转图片
        flip = self.rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        image_data = self.Color_gamut_transform(image, hue, sat, val)

        # 调整目标框坐标
        box_data = self.post_process_box(box, w, nw, iw, nh, ih, dx, dy, flip)

        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    #-----------------------------------#
    #   使用mosaic方法的数据增强
    #-----------------------------------#
    def get_random_data_with_Mosaic(self, annotation_line, input_shape, hue = .1, sat = 1.5, val = 1.5):
        h, w = input_shape
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []
        index = 0

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        # 对每一张(4张)图片进行处理
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB")
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                # 水平翻转
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # 对输入进来的图片进行缩放
            scale = self.rand(scale_low, scale_high)
            nw = int(scale * w)
            nh = int(nw)
            # 不是等比例缩放
            image = image.resize((nw, nh), Image.BICUBIC)
            # new_ar = w / h
            # scale = self.rand(scale_low, scale_high)
            # if new_ar < 1:
            #     nh = int(scale * h)
            #     nw = int(nh * new_ar)
            # else:
            #     nw = int(scale * w)
            #     nh = int(nw / new_ar)
            # image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
            image = self.Color_gamut_transform(image, hue, sat, val)

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', 
                                  (w, h),
                                  (np.random.randint(0, 255), 
                                   np.random.randint(0, 255), 
                                   np.random.randint(0, 255))
                                  )
            # (dx, dy)是左上角坐标，paste可能会损失掉图片的部分区域
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            # 调整目标框坐标
            box_data = self.post_process_box(box, w, nw, iw, nh, ih, dx, dy, flip)
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

        new_image = np.zeros([h, w, 3])
        # y从0到cuty，x从0到cutx的部分放第一张图片
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        # y从cuty到h，x从0到cutx的部分放第二张图片
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        # y从cuty到h，x从cutx到w的部分放第三张图片
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        # y从0到cuty，x从cutx到w的部分放第四张图片
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对框进行进一步的处理
        new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))

        if len(new_boxes) == 0:
            return new_image, []
        if (new_boxes[:, :4] > 0).any():
            return new_image, new_boxes
        else:
            return new_image, []

# DataLoader中collate_fn使用，取数据的时候images和bboxes都以列表的形式取出
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

