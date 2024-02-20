import os
import warnings
import cv2
import pandas as pd
from torch.utils.data import Dataset
import clip

from albumentations import CoarseDropout, RandomGamma, MedianBlur, ToSepia, RandomSnow, RandomToneCurve, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, MotionBlur, RandomRain, RGBShift, RandomFog, Downscale, InvertImg, ColorJitter, Compose, RandomBrightnessContrast, CLAHE, ISONoise, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize
from transforms.trasformations import CustomRandomCrop, IsotropicResize

# Suppress warnings
warnings.simplefilter("ignore")

class ImagesDataset(Dataset):
    def __init__(self, data_path, csv_path, size, set, modal_mode):
        self.data_path = data_path
        self.size = size
        self.mode = modal_mode
        self.df = pd.read_csv(csv_path)
        self.labels = self.df['class'].tolist()

        if self.mode==1: # multimodal image + text
            self.captions = self.df['original_caption'].tolist()

        if set=="train":
            self.transform = self.create_train_transforms(self.size)
        else:
            self.transform = self.create_val_transform(self.size)
        

    def __len__(self):
        return len(self.df)
    
    def create_train_transforms(self, size):
         return Compose([
                OneOf(
                [
                IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC), 
                IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR)
                ], p=1), # equals to CustomRandomCrop(size=self.size, p=1),
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
            return self.__getitem__((idx + 1) % len(self))  # skip sample and move to the next one
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']

        if self.mode==0: # unimodal
            caption = ""
        else: # multimodal
            caption = self.captions[idx]
            if len(caption) > 77:  # check if caption length exceeds 77 (clip max text context length)
                caption = caption[:77]  # truncate caption
            caption = clip.tokenize(caption)

        return (image, caption, label)