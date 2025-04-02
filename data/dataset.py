"""Defines the parking slot dataset for directional marking point detection."""
import json
import os
import os.path
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from data.struct import MarkingPoint

from PIL import Image
import random
import numpy as np
from PIL import ImageEnhance
from util.aug import randomColor, gauss_blur, add_salt_and_pepper_noise

class ParkingSlotDataset(Dataset):
    """Parking slot dataset."""
    def __init__(self, root):
        super(ParkingSlotDataset, self).__init__()
        self.root = root
        self.sample_names = []
        self.image_transform = ToTensor()
        for file in os.listdir(root):
            if file.endswith(".json"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        name = self.sample_names[index]
        image = cv.imread(os.path.join(self.root, name+'.jpg'))

        image = self.image_transform(image)
        marking_points = []
        with open(os.path.join(self.root, name + '.json'), 'r') as file:
            for label in json.load(file):
                marking_points.append(MarkingPoint(*label))
        return image, marking_points

    def __len__(self):
        return len(self.sample_names)
    

class ParkingSlotDatasetEMA(Dataset):
    """Parking slot dataset."""
    def __init__(self, root):
        super(ParkingSlotDatasetEMA, self).__init__()
        self.root = root
        self.sample_names = []
        self.image_transform = ToTensor()
        for file in os.listdir(root):
            if file.endswith(".json"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        name = self.sample_names[index]
        image = cv.imread(os.path.join(self.root, name+'.jpg'))
        image_ema = cv.imread(os.path.join(self.root, name+'.jpg'))

        # 将 OpenCV 图像转换为 PIL 图像
        image_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # 调用 randomColor 函数，对图像进行随机颜色调整
        enhanced_image_pil = randomColor(image_pil)
        # 将 PIL 图像转换回 OpenCV 图像
        enhanced_image = cv.cvtColor(np.array(enhanced_image_pil), cv.COLOR_RGB2BGR)
        image = self.image_transform(enhanced_image) # 强噪声

        image_ema = gauss_blur(image_ema, ksize=(3, 3), sigma=0)
        ema_image = self.image_transform(image_ema) # 弱噪声

        marking_points = []
        with open(os.path.join(self.root, name + '.json'), 'r') as file:
            for label in json.load(file):
                marking_points.append(MarkingPoint(*label))
        return image, ema_image, marking_points

    def __len__(self):
        return len(self.sample_names)

class ParkingSlotDatasetWithLabel(Dataset):
    """Parking slot dataset."""
    def __init__(self, root):
        super(ParkingSlotDatasetWithLabel, self).__init__()
        self.root = root
        self.sample_names = []
        self.image_transform = ToTensor()
        for file in os.listdir(root):
            if file.endswith(".json"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        name = self.sample_names[index]
        image = cv.imread(os.path.join(self.root, name+'.jpg'))

        image = self.image_transform(image) 

        marking_points = []
        with open(os.path.join(self.root, name + '.json'), 'r') as file:
            for label in json.load(file):
                marking_points.append(MarkingPoint(*label))
        return image, marking_points

    def __len__(self):
        return len(self.sample_names)
    
class ParkingSlotDatasetWithoutLabel(Dataset):
    """Parking slot dataset."""
    def __init__(self, root):
        super(ParkingSlotDatasetWithoutLabel, self).__init__()
        self.root = root
        self.sample_names = []
        self.image_transform = ToTensor()
        for file in os.listdir(root):
            if file.endswith(".jpg"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
            name = self.sample_names[index]
            image_week = cv.imread(os.path.join(self.root, name+'.jpg'))
            image_strong = cv.imread(os.path.join(self.root, name+'.jpg'))

            image_week = add_salt_and_pepper_noise(image_week, noise_ratio=0.003)
            week_image = self.image_transform(image_week) # 椒盐噪声


            image_strong = add_salt_and_pepper_noise(image_strong, noise_ratio=0.003)
            strong_image = self.image_transform(image_strong) # 椒盐噪声

            return week_image, strong_image
    
    def __len__(self):
        return len(self.sample_names)

