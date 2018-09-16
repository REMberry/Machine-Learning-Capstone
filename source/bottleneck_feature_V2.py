from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
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

# dimensions of our images.
img_width, img_height = 150, 150

file_path = 'saved_models/bottleneck_fc_model_best1.h5'
top_model_weights_path = 'saved_models/bottleneck_fc_model1.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
batch_size = 64
nb_train_samples = 976
nb_validation_samples = 100


def save_bottlebeck_features(train_tensors, train_targets, valid_tensors, test_tensors):

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('validation data prediction')
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.2,
        shear_range=0.2,  # set range for random shear
        zoom_range=0.2,  # set range for random zoom
        channel_shift_range=0.0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0) 

    datagen.fit(train_tensors)

    generator_top = datagen.flow(  
         train_tensors, train_targets ,
         #target_size=(img_width, img_height),  
         batch_size=batch_size,  
         shuffle=True)  

   
    predict_size_train = (nb_train_samples // batch_size)
   
    bottleneck_features_train = model.predict_generator(     
    generator_top, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    train_labels = generator_top.classes  
    train_labels = to_categorical(train_labels, num_classes=10)  


    bottleneck_features_train = model.predict(train_tensors)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    bottleneck_features_valid = model.predict(valid_tensors)
    np.save('bottleneck_features_valid.npy', bottleneck_features_valid)

    bottleneck_features_test = model.predict(test_tensors)
    np.save('bottleneck_features_test.npy', bottleneck_features_test)



def train_top_model(train_targets, valid_targets, test_targets):

    train_data = np.load('bottleneck_features_train.npy')
    valid_data = np.load('bottleneck_features_valid.npy')
    test_data = np.load('bottleneck_features_test.npy')

    model = Sequential()
    model.add(BatchNormalization(input_shape=train_data.shape[1:]))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    #Set Opt
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    model.summary

    checkpointer = ModelCheckpoint(filepath=file_path, 
                               verbose=1, save_best_only=True)

    history = model.fit(train_data, train_targets,
              #steps_per_epoch = 100,
              epochs=200,
              batch_size=64,
              shuffle = True,
              verbose=1,
              validation_data=(valid_data, valid_targets),
              callbacks=[checkpointer])
    model.save_weights(top_model_weights_path)




    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

save_bottlebeck_features(train_tensors,train_targets, valid_tensors, test_tensors)

print('train top model')
train_top_model(train_targets, valid_targets, test_targets)