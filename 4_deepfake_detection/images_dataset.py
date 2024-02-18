import torch
from torch.utils.data import Dataset
import cv2 
import numpy as np
from albumentations import Cutout, CoarseDropout, RandomGamma, MedianBlur, ToSepia, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, RandomBrightness, MotionBlur, RandomRain, RGBShift, RandomFog, RandomContrast, Downscale, InvertImg, RandomContrast, ColorJitter, Compose, RandomBrightnessContrast, CLAHE, ISONoise, JpegCompression, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize

from transforms.albu import IsotropicResize, CustomRandomCrop

class ImagesDataset(Dataset):
    def __init__(self, images, labels, image_size, mode = 'train'): #captions
        self.x = images
        #self.captions = clip.tokenize(captions)
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = images.shape[0]
    
    def create_train_transforms(self, size):
         return Compose([
                CustomRandomCrop(size=size, p=1),
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

    """def create_train_transforms(self, size):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )"""

    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])

    def __getitem__(self, index):
        image = np.asarray(self.x[index])
        #caption = np.asarray(self.captions[index])

        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)
        image = transform(image=image)['image']
        return torch.tensor(image).float(), self.y[index]



    def __len__(self):
        return self.n_samples