import os
import random

xmlfilepath = './VOCdevkit/VOC2007/Annotations'
saveBasePath = "./VOCdevkit/VOC2007/ImageSets/Main/"

# trainval = train + val
# train + val 总数量比例（占图片总数）
trainval_percent = 0.75

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    # endswith用于判断字符串是否以指定元素结尾
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)
list = range(num)
# 图片总数 * (train + val 比例) = train + val 总数
tv = int(num * trainval_percent)

trainval = random.sample(list, tv)

print("train and val size", tv)
print("test size", num - tv)

# 建立四类数据集txt文件
ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
 
for i in list:  
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
    else:
        ftest.write(name)
  
ftrainval.close()
ftest.close()
