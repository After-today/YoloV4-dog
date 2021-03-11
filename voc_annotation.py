import xml.etree.ElementTree as ET
from os import getcwd

# sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = ['trainval', 'test']

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ['dog']

def convert_annotation(image_id, list_file):
    # 打开image_id对应的xml文件
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml' % image_id)
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得根节点
    root = tree.getroot()

    # 以object元素为根节点创建树迭代器
    for obj in root.iter(tag = 'object'):
        # difficult 目标是否难以识别（0表示容易 1表示困难）
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        # 如果没有这个class或目标难以识别，则该目标不计入数据集
        if cls not in classes or int(difficult) == 1:
            continue
        
        # 获得类别的索引
        cls_id = classes.index(cls)
        # 获得标注框的坐标信息
        xmlbox = obj.find('bndbox')
        # 解析坐标信息（左上角和右下角坐标）
        b = (int(xmlbox.find('xmin').text), 
             int(xmlbox.find('ymin').text), 
             int(xmlbox.find('xmax').text), 
             int(xmlbox.find('ymax').text))
        # ' ' + 'xmin,ymin,xmax,ymax' + ',' + 类别号
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

# 获得当前工作路径
wd = getcwd()

for image_set in sets:
    # 打开main文件夹下数据集的txt文件（存储图片序号image_ids）
    image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % image_set).read().strip().split('\n')
    # 创建当前目录下对应数据集的txt文件（存储标注信息）
    list_file = open('%s.txt' % image_set, 'w')

    for image_id in image_ids:
        # 写入每张图片image_id的绝对路径
        list_file.write('%s/VOCdevkit/VOC2007/JPEGImages/%s.jpg' % (wd, image_id))
        # 写入image_id对应的 ' ' + 'xmin,ymin,xmax,ymax' + ',' + 类别号
        convert_annotation(image_id, list_file)
        # 一张图片image_id结束，换行
        list_file.write('\n')
    list_file.close()
