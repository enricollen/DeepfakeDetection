import os
import warnings
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Normalize

from albumentations import CoarseDropout, RandomGamma, MedianBlur, ToSepia, RandomSnow, RandomToneCurve, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, MotionBlur, RandomRain, RGBShift, RandomFog, Downscale, InvertImg, ColorJitter, Compose, RandomBrightnessContrast, CLAHE, ISONoise, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, RandomCrop, Resize
from transforms.trasformations import CustomRandomCrop, IsotropicResize

# Suppress warnings
warnings.simplefilter("ignore")
os.environ["OPENCV_LOG_LEVEL"]="SILENT"

class ImagesDataset(Dataset):
    def __init__(self, data_path, csv_path, size, tokenizer, set, modal_mode):
        self.data_path = data_path
        self.size = size
        self.set = set
        self.mode = modal_mode
        self.tokenizer = tokenizer
        self.df = pd.read_csv(csv_path)
        self.labels = self.df['class'].tolist()

        if self.mode==1: # multimodal image + text
            self.captions = self.df['original_caption'].tolist()

        if self.set=="train":
            self.transform = self.create_train_transforms(self.size)
        else:
            self.transform = self.create_val_test_transform(self.size)
        

    def __len__(self):
        return len(self.df)
    
    def create_train_transforms(self, size):
         return Compose([
                CustomRandomCrop(size=self.size, p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
                OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur()], p=0.1),
                OneOf([HorizontalFlip(), InvertImg()], p=0.5),
                OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue(), RandomToneCurve()], p=0.5),
                OneOf([RGBShift(), ColorJitter()], p=0.1),
                OneOf([MultiplicativeNoise(), ISONoise(), GaussNoise()], p=0.3),
                OneOf([RandomFog(), RandomRain(), RandomSnow(), RandomSunFlare()], p=0.05),
                CoarseDropout(p=0.1),
                RandomShadow(p=0.1),
                RandomGamma(p=0.1),
                CLAHE(p=0.05),
                ToGray(p=0.2),
                ToSepia(p=0.05),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                Resize(224,224)
            ])
    
    def create_val_test_transform(self, size):
        return Compose([
            CustomRandomCrop(size=self.size, p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            Resize(224,224)
        ])
    
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['id']
        label = self.labels[idx]  
        image_path = os.path.join(self.data_path, image_name + '.jpg')

        image = cv2.imread(image_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))  # skip sample and move to the next one
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image) / 255.0
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        if self.mode == 0:  # unimodal
            caption = ""
        else:  # multimodal
            caption = self.captions[idx]
            caption = self.tokenizer.encode_plus(
                caption,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        
        if self.set == "test":
            return (image_name, image, caption, label)
        else: # train and val do not need the image name for evaluating performance
            return (image, caption, label)