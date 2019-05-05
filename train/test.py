import sys
import numpy
import caffe
import cv2
import numpy as np
import skimage
from skimage import transform

WEIGHTS_FILE = 'freq_regression_iter_10000.caffemodel'
DEPLOY_FILE = 'deploy.prototxt'
IMAGE_SIZE = (100, 100)
MEAN_VALUE = 128

caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)
#net.blobs['data'].reshape(1, 3, *IMAGE_SIZE)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2,1,0)) 
#transformer.set_mean('data', numpy.array([MEAN_VALUE]))

image_list = sys.argv[1]

batch_size = net.blobs['data'].data.shape[0]
with open(image_list, 'r') as f:
    i = 0
    filenames = []
    for line in f.readlines():
        filename = line[:-1]
        #filename = '/Users/lvliang/ml/Wechat_AutoJump/cnn_coarse_to_fine/data_provider/data/exp_03/state_181_res_h_620_w_324.png'
        filenames.append(filename)
#        image = cv2.imread(filename)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = caffe.io.load_image(filename, True)[320:-320,40:-40,0].astype(np.float32)
        #print image
        #image = cv2.imread(filename)[320:-320,:,:]
        #image = cv2.resize(image,IMAGE_SIZE)
        #image = image.reshape((3, )+IMAGE_SIZE) 
        #image = image.astype(np.float32)
        image = transform.resize(image,(100,100))
        #image = image.reshape(3,100,100)
        #print image.shape
        #image = image[-320:320,:,:]
        #image = caffe.io.resize_image(image, (3,)+IMAGE_SIZE)
        #image -= 128;
        transformed_image = transformer.preprocess('data', image)
        #f = open("out_test.txt","w")
        #print >>f,transformed_image
        #f.close
        #transformed_image -= MEAN_VALUE
        #print transformed_image
        net.blobs['data'].data[...] = transformed_image
        i += 1

        if i == batch_size:
            output = net.forward()
            freqs = output['pred']

            for filename, (fx, fy) in zip(filenames, freqs):
                print('Predicted frequencies for {} is {:.7f} and {:.7f}'.format(filename, fx * 1280,
                    fy * 720))
                
            i = 0
            filenames = []
