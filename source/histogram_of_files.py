from __future__ import print_function
import keras
from keras.models import Model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from sklearn.datasets import load_files
from keras.callbacks import ModelCheckpoint  
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

import numpy as np
import os
from keras.utils import np_utils
from tqdm import tqdm
from keras.preprocessing import image   
from keras import applications
import pylab as plt

import itertools
from keras.preprocessing import image      
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pylab as plt
import plotly.plotly as py
import plotly.tools as tls

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(150, 150))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def load_dataset(path):
    data = load_files(path)
    mushroom_files = np.array(data['filenames'])
    mushroom_targets = np_utils.to_categorical(np.array(data['target']), 10)
    return mushroom_files, mushroom_targets



train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

#train_tensors = paths_to_tensor(train_files).astype('float32')/255
#valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
#test_tensors = paths_to_tensor(test_files).astype('float32')/255

number_of_train = np.zeros(train_targets.shape[0])
number_of_valid = np.zeros(valid_targets.shape[0])
number_of_test = np.zeros(test_targets.shape[0])

for i in range(train_targets.shape[0]):
    number_of_train[i] = np.argmax(train_targets[i])+1

for i in range(valid_targets.shape[0]):
    number_of_valid[i] = np.argmax(valid_targets[i])+1

for i in range(test_targets.shape[0]):
    number_of_test[i] = np.argmax(test_targets[i])+1


#fig = plt.figure()
#n, bins, patches = plt.hist([number_of_files])
#plotly_fig = tls.mpl_to_plotly( fig )
#py.iplot(plotly_fig, filename='mpl-multi-histogram')

n_bins = 10

fig = plt.figure()
datasets = ['train', 'validation', 'test']

plt.hist([number_of_train,number_of_valid,number_of_test ] , normed=False, bins=10, label=datasets)
plt.legend(prop={'size': 10})
plt.title('histogram of dataset')
plt.ylabel('number of mushrooms in dataset')

plt.show()
#plt.hist(number_of_files, n_bins, normed=1, histtype='bar', color=colors, label=colors)
#plt.legend(prop={'size': 10})
#plt.set_title('bars with legend')
#plt.show()

print('end')