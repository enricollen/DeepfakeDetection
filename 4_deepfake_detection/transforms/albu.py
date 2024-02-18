import random

import cv2
import numpy as np
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.crops.functional import crop


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]

    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down

    img = img.astype('uint8')
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")



class CustomRandomCrop(DualTransform):
    def __init__(self, size, p=0.5) -> None:
        super(CustomRandomCrop, self).__init__(p=p)
        self.size = size
        self.prob = p

    def apply(self, img, copy=True, **params):
        # Check if the image dimensions are smaller than self.size
        if img.shape[0] < self.size or img.shape[1] < self.size:
            x = random.rand()
            if x < 0.3:
                transform = IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR)

            elif x > 0.3 and x < 0.6:
                transform = IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR)

            else: 
                transform = IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC)
            
            # Apply the chosen transformation
            img = transform(image=img)["image"]
        
        return img