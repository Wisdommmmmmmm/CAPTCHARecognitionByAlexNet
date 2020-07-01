import string
import requests
import time
from io import BytesIO
from PIL import Image
import os
import numpy as np
from PIL import Image
import torch as t
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import transforms as T

threshold = 125
LUT = threshold*[0] + (256-threshold)*[1]
charset = '34678abcdehijknprtuvxy'

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
    #img = img.point(LUT, "1")
    res = np.array(img, dtype=np.float32)
    res = t.from_numpy(res)
    res = res.unsqueeze(0)
    return res

def Test(img):
    img = img.resize((227, 227))
    img = ToTensor(img)
    input = img.unsqueeze(0)
    input = Variable(input)
    model=t.load('MyAlexNet.pth')
    model.eval()                  
    output = model(input)
    kk = 0
    for k in range(21):
        if output[0][k + 1] > output[0][kk]:
            kk = k + 1
    return charset[kk]

def handle_captcha(capt):
    #灰度化
    capt_gray = capt.convert("L")
    #二值化
    capt_b = capt_gray.point(LUT, "1")
    for i in range(5):
        x = 0 + i * 15
        y = 0
        capt_char = capt_b.crop((x, y, x + 15, y + 33))
        print(Test(capt_char))

capt = Image.open('1.png')
handle_captcha(capt)
