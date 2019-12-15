import numpy as np

from models import ALOCC_Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import backend as K
import os
from keras.losses import binary_crossentropy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


self =ALOCC_Model(dataset_name='mnist', input_height=32,input_width=32)
self.adversarial_model.load_weights('./checkpoint/ALOCC_Model_30.h5')


X_train = np.load("lesion_test_25_100.npy")
X_train = X_train[:,:,:,np.newaxis]

def test_reconstruction(label, data_index = 0):
    # specific_idx = np.where(y_train == label)[0]
    if data_index >= len(X_train):
        data_index = 0
    datalist = X_train
    for i in range(len(datalist)):
        data = X_train[i:i+1]
        model_predicts = self.adversarial_model.predict(data)
        
        # print(model_predicts_latentspace[0])
        #fig= plt.figure(figsize=(8, 8))
        #columns = 1
        #rows = 2
        #fig.add_subplot(rows, columns, 1)
        input_image = data.reshape((32, 32))
        reconstructed_image = model_predicts[0].reshape((32, 32))
        #plt.title('Input')
        #plt.imshow(input_image, label='Input')
        #fig.add_subplot(rows, columns, 2)
        #plt.title('Reconstruction')
        #plt.imshow(reconstructed_image, label='Reconstructed')
        #plt.show()
        # Compute the mean binary_crossentropy loss of reconstructed image.
        y_true = K.variable(reconstructed_image)
        y_pred = K.variable(input_image)
        error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
        print(error+1-model_predicts[1][0][0])



test_reconstruction(4)
