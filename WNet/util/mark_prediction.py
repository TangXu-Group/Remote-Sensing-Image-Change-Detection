import cv2
import os
from PIL import Image
import random
Image.MAX_IMAGE_PIXELS = None
import numpy as np

if __name__ == '__main__':
    dir = os.path.dirname(__file__)
    dir_label = os.path.join(dir, 'label.png')
    dir_pred = os.path.join(dir, 'prediction.png')
    label = Image.open(dir_label)
    pred = Image.open(dir_pred)
    label_rgb = label.convert('RGB')
    label_np = np.array(Image.open(dir_label))
    pred_np = np.array(Image.open(dir_pred))
    label_rgb_np = np.array(label_rgb)
    for i in range(256):
        for j in range(256):
            if label_np[i][j] == 255 and pred_np[i][j] == 0:
                label_rgb_np[i][j] = [255, 0, 0]
            elif label_np[i][j] == 0 and pred_np[i][j] == 255:
                label_rgb_np[i][j] = [0, 255, 0]
            else:
                pass
    img = Image.fromarray(label_rgb_np)
    img.save("marked_prediction.png")









    
