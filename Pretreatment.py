import string
import requests
import time
from io import BytesIO
from PIL import Image
from pytesseract import image_to_string
import os
import numpy as np

#验证码字符集
charset = '34678abcdehijknprtuvxy'
#验证码网络
captcha_url = "https://epass.icbc.com.cn/servlet/com.icbc.inbs.person.servlet.Verifyimage2?disFlag=2&randomKey=1580537733595658615&width=70&height=28&appendRandom=1580537819345"
#验证码保存路径
captcha_path = "C:/Users/Lenovo/Desktop/graduation design/captcha/test/"
if not os.path.exists(captcha_path):
    os.mkdir(captcha_path)
threshold = 105
LUT = threshold*[0] + (256-threshold)*[1]
l = {}

#爬取验证码并转换成image对象
def get_captcha():
    captcha_origin = requests.get(captcha_url)
    f = BytesIO(captcha_origin.content)
    captcha_res = Image.open(f)
    return captcha_res

#验证码预处理并保存
def handle_captcha():
    capt = get_captcha()
    capt.save("1.jpg")
    #灰度化
    capt_gray = capt.convert("L")
    capt_gray.save("2.jpg")
    #二值化
    capt_b = capt_gray.point(LUT, "1")
    capt_b.save("3.jpg")
    #切边框
    capt_bx = capt_b.crop((1, 1, 69, 27))
    '''
    text = image_to_string(capt_bx, lang = 'eng', config='--psm 8 -c tessedit_char_whitelist='+charset)
    if len(text) != 4:
        return
    '''
    for i in range(4):
        x = 2 + i * 16
        y = 0
        capt_char = capt_bx.crop((x, y, x + 16, y + 25))
        capt_char.save(str(x+100)+".jpg")
        #path = captcha_path + text[i] + "/"
        #capt_char.save(path+str(l[text[i]]).zfill(6)+".jpg")
        #l[text[i]] += 1

def rename(path):
    i = 0
    '该文件夹下所有的文件（包括文件夹）'
    FileList = os.listdir(path)
    '遍历所有文件'
    for files in FileList:
        '原来的文件路径'
        oldDirPath = os.path.join(path, files)
        '如果是文件夹则递归调用'
        if os.path.isdir(oldDirPath):
            rename(oldDirPath)
            continue
        '文件名'
        fileName = os.path.splitext(files)[0]
        '文件扩展名'
        fileType = os.path.splitext(files)[1]
        '新的文件路径'
        newDirPath = os.path.join(path, str(i).zfill(6) + fileType)
        '重命名'
        if oldDirPath == newDirPath:
            i += 1
            continue
        os.rename(oldDirPath, newDirPath)
        i += 1
if(__name__=="__main__"):
    rename(captcha_path)
    '''
    for i in charset:
        path = captcha_path + i + "/"
        if not os.path.exists(path):
            os.mkdir(path)
        l[i] = len([lists for lists in os.listdir(path) if os.path.isfile(os.path.join(path, lists))])
    '''
    #handle_captcha()
    #for i in range(1000):
        #handle_captcha()
