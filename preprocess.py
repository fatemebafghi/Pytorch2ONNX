import numpy as np
import cv2
from torchvision import transforms
import torchvision.transforms.functional as function

def classifier_preprocess(image):


    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [SquarePad(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
         ]
    )
    image = transform(image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('f')
    return image




class SquarePad2:
    def __call__(self, image):
        _, w, h = image.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        return cv2.copyMakeBorder(image, hp, vp, hp, vp, cv2.BORDER_CONSTANT, value=(0, 0, 0))

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return function.pad(image, padding, 0, 'constant')

