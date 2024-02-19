import os
import warnings
import cv2
import pandas as pd
from torch.utils.data import Dataset

from albumentations import Cutout, CoarseDropout, RandomGamma, MedianBlur, ToSepia, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, RandomBrightness, MotionBlur, RandomRain, RGBShift, RandomFog, RandomContrast, Downscale, InvertImg, RandomContrast, ColorJitter, Compose, RandomBrightnessContrast, CLAHE, ISONoise, JpegCompression, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize
from transforms.albu import CustomRandomCrop, IsotropicResize

# Suppress warnings
warnings.simplefilter("ignore")

class LazyImagesDataset(Dataset):
    def __init__(self, data_path, csv_path, size, mode):
        self.data_path = data_path
        self.size=size
        self.df = pd.read_csv(csv_path)
        self.labels = self.df['class'].tolist()
        if mode=="train":
            self.transform = self.create_train_transforms(self.size)
        else:
            self.transform = self.create_val_transform(self.size)
        

    def __len__(self):
        return len(self.df)
    
    def create_train_transforms(self, size):
         return Compose([
                IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                #Resize(height=size, width=size),
                ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
                OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur()], p=0.1),
                OneOf([HorizontalFlip(), InvertImg()], p=0.5),
                OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(), HueSaturationValue()], p=0.5),
                OneOf([RGBShift(), ColorJitter()], p=0.1),
                OneOf([MultiplicativeNoise(), ISONoise(), GaussNoise()], p=0.3),
                OneOf([Cutout(), CoarseDropout()], p=0.1),
                OneOf([RandomFog(), RandomRain(), RandomSunFlare()], p=0.02),
                RandomShadow(p=0.05),
                RandomGamma(p=0.1),
                CLAHE(p=0.05),
                ToGray(p=0.2),
                ToSepia(p=0.05),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ])
    
    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])
    
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['id']
        label = self.labels[idx]  
        image_path = os.path.join(self.data_path, image_name + '.jpg')

        image = cv2.imread(image_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))  # Skip sample and move to the next one
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']

        return (image, label)