from PIL import Image
import random
import numpy as np
from PIL import ImageEnhance
import cv2

def randomColor(image):
    # 随机生成0,1来随机确定调整哪个参数，可能会调整饱和度，也可能会调整图像的饱和度和亮度
    saturation = random.randint(0, 1)
    brightness = random.randint(0, 1)
    contrast = random.randint(0, 1)
    sharpness = random.randint(0, 1)

    # 当三个参数中一个参数为1，就可执行相应的操作
    if random.random() < saturation:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return image

# 高斯模糊
def gauss_blur(img, ksize, sigma):
    # 外部调用传入正整数即可,在这里转成奇数
    k_list = list(ksize)
    kw = (k_list[0] * 2) + 1
    kh = (k_list[1] * 2) + 1
    resultImg = cv2.GaussianBlur(img, (kw, kh), sigma)
    return resultImg

def add_salt_and_pepper_noise(image, noise_ratio):
    h, w, _ = image.shape
    num_noise_pixels = int(noise_ratio * h * w)

    # 在随机位置生成椒盐噪声像素
    coords = np.random.randint(0, high=max(h, w), size=(num_noise_pixels, 2))
    for coord in coords:
        if np.random.random() < 0.5:
            image[coord[0], coord[1], :] = 0  # 将噪声像素设置为黑色
        else:
            image[coord[0], coord[1], :] = 255  # 将噪声像素设置为白色



    return image

def add_gaussian_noise(image, mean, std_dev):
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

