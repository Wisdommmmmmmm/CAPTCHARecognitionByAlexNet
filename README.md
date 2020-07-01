# 基于卷积神经网络的文本验证码识别
深度学框架：PyTorch  
卷积神经网络模型：AlexNet  
AlexNet.py:训练网络模型  
CaptchaRecognize.py:利用训练好的网络模型进行验证码识别  
Plot.py:画出损失函数值和迭代次数的关系图  
Pretreatment.py:利用爬虫技术从工商银行官网爬取验证码图像，并对验证码图像进行灰度化、二值化、切割等一系列预处理后分为训练集和测试集  
Test.py:测试并计算训练好的模型在测试集上的识别率  
实验结果：测试集识别率达99.76%，并且能对类工行验证码邮政储蓄银行验证码进行识别，具有泛化能力。  
