import numpy as np
import cv2
from PIL import Image
import numpy as np
import os
import subprocess
import random 

def load_data(data_dir, file_names):
    if not os.path.exists(data_dir):
        subprocess.call('source download_dataset.sh')

    a_img = np.empty((0,256,256), dtype=np.float32)
    b_img = np.empty((0,256,256,3), dtype=np.float32)
    
    for i, name in enumerate(file_names):
        img = np.asarray(Image.open(os.path.join(data_dir, name)))
        w = img.shape[1]//2
        aimg = cv2.cvtColor(img[:, :w,:], cv2.COLOR_RGB2GRAY)/255
        bimg = img[:, w:, :]/255
        a_img = np.append(a_img, np.array([aimg]), axis=0)
        b_img = np.append(b_img, np.array([bimg]), axis=0)

    a_img = np.expand_dims(a_img, axis=-1)
    return a_img, b_img

def generator(batch_size, data_dir):
    file_names = os.listdir(data_dir)

    def gen():
        
        while True:
            selected_name = random.sample(file_names, batch_size)
            x_, y_ = load_data(data_dir, selected_name)
            yield x_, y_
    return gen
