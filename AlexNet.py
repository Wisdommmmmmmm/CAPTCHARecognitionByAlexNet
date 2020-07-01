# coding:utf8
from PIL import Image
import torch as t
import torch.nn as nn
import numpy as np
import os
import random
from torch.autograd import Variable
from torchvision import transforms as T

charset = '34678abcdehijknprtuvxy'
captcha_path = "C:/Users/Lenovo/Desktop/graduation design/captcha/train/"
threshold = 105
LUT = threshold*[0] + (256-threshold)*[1]

class AlexNet(nn.Module):
    '''
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''

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


_loss = []
l = {}
optimizer = t.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.001)
scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

T = 3000
for i in charset:
    path = captcha_path + i + "/"
    if not os.path.exists(path):
        os.mkdir(path)
    l[i] = len([lists for lists in os.listdir(path) if os.path.isfile(os.path.join(path, lists))])
    T = min(T, l[i])
#标签
label = t.tensor([0])
for i in range(22):
    if i == 0:
        continue
    label = t.cat((label, t.tensor([i])), 0)

for x in range(T):
    img = Image.open(captcha_path + charset[0] + '/' + str(x).zfill(6) + '.jpg')
    img = img.resize((227, 227))
    img = ToTensor(img)
    input = img.unsqueeze(0)
    for i in range(22):
        if i == 0:
            continue
        img = Image.open(captcha_path + charset[i] + '/' + str(x).zfill(6) + '.jpg')
        img = img.resize((227, 227))
        img = ToTensor(img)
        input = t.cat((input, img.unsqueeze(0)), 0)
    input = Variable(input)
    target = Variable(label)
    optimizer.zero_grad()  #把梯度置零，也就是把loss关于weight的导数变成0.
    output = model(input)  #给网络一个输入input，得到一个二维的输出output，
    criterion = t.nn.CrossEntropyLoss()  #交叉熵损失，
    loss = criterion(output, target)
    _loss.append(float(loss))
    print(x)
    print(float(loss))
    loss.backward()
    optimizer.step()
    scheduler.step()

t.save(model, captcha_path+'MyAlexNet.pth')
fb = open(captcha_path+'losslist.txt', 'a')
for i in _loss:
    fb.write(str(i))
    fb.write('\n')
fb.close()