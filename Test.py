# coding:utf8
from PIL import Image
import torch as t
import torch.nn as nn
import numpy as np
import os
import random
import time
from torch.autograd import Variable
from torchvision import transforms as T
import matplotlib.pyplot as plt

charset = '34678abcdehijknprtuvxy'
captcha_path = "C:/Users/Lenovo/Desktop/graduation design/captcha/test/"
threshold = 105
LUT = threshold*[0] + (256-threshold)*[1]
unloader = T.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    #print(image.shape)
    image = unloader(image)
    return image

class AlexNet(nn.Module):
    def __init__(self, num_classes=22):
        super(AlexNet, self).__init__()
        self.model_name = 'alexnet'
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        plt.figure()
        for i in range(1, 257):
            plt.subplot(16, 16, i)
            plt.imshow(tensor_to_PIL(x[0][i-1]))
            plt.xticks([])
            plt.yticks([])
        plt.show()
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def ToTensor(img):
    img = img.point(LUT, "1")
    res = np.array(img, dtype=np.float32)
    res = t.from_numpy(res)
    res = res.unsqueeze(0)
    return res

'''
# 直接加载网络：
model = AlexNet()

'''
# 加载本地模型：
model=t.load(captcha_path+'MyAlexNet.pth')
model.eval()

_loss = []
l = {}
'''
for i in charset:
    path = captcha_path + i + "/"
    if not os.path.exists(path):
        os.mkdir(path)
    l[i] = len([lists for lists in os.listdir(path) if os.path.isfile(os.path.join(path, lists))])
'''
tot = 0
tt = 0
for i in range(1):
    for j in range(1):
        img = Image.open(captcha_path + charset[i] + '/' + str(j).zfill(6) + '.jpg')
        img = img.resize((227, 227))
        img = ToTensor(img)
        input = img.unsqueeze(0)
        input = Variable(input)
        output = model(input)
        for i in range(10):
            input = model.features[i](input)
            if i == 2:
                plt.figure()
                for i in range(1, 65):
                    plt.subplot(8, 8, i)
                    plt.imshow(tensor_to_PIL(input[0][i - 1]))
                    plt.xticks([])
                    plt.yticks([])
                plt.show()
        kk = 0
        for k in range(21):
            if output[0][k+1] > output[0][kk] :
                kk = k+1
        tot += 1
        if kk == i :
            tt += 1
print(tt / tot)
