import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import skimage
from PIL import Image
from skimage import transform
import cv2

IMAGE_SIZE = (100,100)
MEAN_VALUE = 128

filename = sys.argv[1]
setname, ext = filename.split('.')

with open(filename, 'r') as f:
    lines = f.readlines()

np.random.shuffle(lines)

sample_size = len(lines)
imgs = np.zeros((sample_size, 1,) + IMAGE_SIZE, dtype=np.float32)
freqs = np.zeros((sample_size, 2), dtype=np.float32)

h5_filename = '{}.h5'.format(setname)
with h5py.File(h5_filename, 'w') as h:
    for i, line in enumerate(lines):
        image_name, fx, fy = line[:-1].split()
        img = plt.imread(image_name)[320:-320,40:-40,0].astype(np.float32)
        img = transform.resize(img, IMAGE_SIZE)
        x, y = img.shape[0:2]
        img = img.reshape(1,100,100)
        img *= 255;
        imgs[i] = img
        freqs[i] = [float(fx)/1280, float(fy)/720]
        if (i+1) % 100 == 0:
            print('Processed {} images!'.format(i+1))
    h.create_dataset('data', data=imgs)
    h.create_dataset('freq', data=freqs)

with open('{}_h5.txt'.format(setname), 'w') as f:
    f.write(h5_filename)
