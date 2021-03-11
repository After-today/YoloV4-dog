#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from PIL import Image

from yolo import YOLO

# YOLO.get_defaults('classes_path')
# YOLO.get_defaults('model_path')

yolo = YOLO()


while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()


#-------------------------------------#
#       对多张图片预测并保存
#-------------------------------------#
imgs_path = []
imgs_lines = open('./VOCdevkit/VOC2007/ImageSets/Main/test.txt')
for line in imgs_lines:
    imgs_path.append(line.strip())

# 加载多张图像
imgs = './VOCdevkit/VOC2007/JPEGImages/'
img_list = []
for i in range(len(imgs_path)):
    img = imgs_path[i] + '.jpg'
    img_list.append(img)

# 多张图片预测并保存
for i in range(len(img_list)):
    image = Image.open(imgs + img_list[i])
    r_image = yolo.detect_image(image)
    r_image.save('./test_result/' + img_list[i])
