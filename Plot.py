import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
path = "C:/Users/Lenovo/Desktop/graduation design/captcha/train/losslist.txt"
x = []
y = []
n = 0
for line in open(path, 'r'):
    t = float(line)
    if t > 10 :
        continue
    n += 1
    x.append(n)
    y.append(t)
plt.plot(x, y)
plt.show()