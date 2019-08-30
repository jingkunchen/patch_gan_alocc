'''
@Description: In User Settings Edit
@Author: Jingkun Chen
@Date: 2019-08-12 20:43:20
@LastEditTime: 2019-08-29 16:05:38
@LastEditors: Jingkun Chen
'''
import numpy as np

from models import ALOCC_Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import binary_crossentropy
import scipy
from utils import *


self =ALOCC_Model(dataset_name='mnist', input_height=28,input_width=28)
self.adversarial_model.load_weights('./checkpoint/ALOCC_Model_4.h5')
(X_train, y_train), (_, _) = mnist.load_data()
print("X_train:",X_train.shape)
X_train = X_train / 255.

def test_reconstruction(label, data_index = 0):
    specific_idx = np.where(y_train != label)[0]
    if data_index >= len(X_train):
        data_index = 0
    datalist = X_train[specific_idx].reshape(-1, 28, 28, 1)[:1000,:,:,:]
    reconlist = []
    inputlist = []
    for i in range(len(datalist)):
        data = datalist[i:i+1]
        model_predicts = self.adversarial_model.predict(data)
        
        # print(model_predicts_latentspace[0])
#        
        columns = 1
        rows = 2
#        
        input_image = data.reshape((28, 28))
        reconstructed_image = model_predicts[0].reshape((28, 28))
        inputlist.append(input_image)
        reconlist.append(reconstructed_image)
        # fig= plt.figure(figsize=(8, 8))
        # fig.add_subplot(rows, columns, 1)
        # plt.title('Input')
        # plt.imshow(input_image, label='Input')
        # fig.add_subplot(rows, columns, 2)
        # plt.title('Reconstruction')
        # plt.imshow(reconstructed_image, label='Reconstructed')
        # plt.show()
        # Compute the mean binary_crossentropy loss of reconstructed image.
        y_true = K.variable(reconstructed_image)
        y_pred = K.variable(input_image)
        error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
        # print('Reconstruction loss, Discriminator Output, Discriminator latentspace Output:', error,model_predicts[1][0][0].mean())
        print(error+ 1- model_predicts[1][0][0].mean())
    reconlist = np.asarray(reconlist)
    inputlist = np.asarray(inputlist)
    print("reconlist:",reconlist.shape)
    scipy.misc.imsave('./{}/test_input_samples.jpg'.format(self.sample_dir), montage(inputlist[:25,:,:]))
    scipy.misc.imsave('./{}/test_output_samples.jpg'.format(self.sample_dir), montage(reconlist[:25,:,:]))

test_reconstruction(1)
