'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pylab as plt

# Set Variables
batch_size = 32
num_classes = 10
epochs = 1
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'modelFromUdacityV3-2.h5'
file_path = 'saved_models/weights.best.from_V3-2.hdf5'

# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

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

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

print(train_tensors.shape)

model = Sequential()

model.add(BatchNormalization(input_shape=(150, 150, 3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='lecun_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='lecun_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='lecun_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='lecun_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='lecun_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

checkpointer = ModelCheckpoint(filepath=file_path, 
                               verbose=1, save_best_only=True)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(train_tensors, train_targets,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(valid_tensors, valid_targets),
              shuffle=True,
              callbacks=[checkpointer], verbose=1
              )
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.15,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.15,
        shear_range=0.1,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        horizontal_flip=True,  # randomly flip images
        ) 

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # Fit the model on the batches generated by datagen.flow().

    datagen.fit(train_tensors)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(train_tensors, train_targets,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(valid_tensors, valid_targets),
                        shuffle = True,
                        steps_per_epoch= train_tensors.shape[0] // batch_size,
                        callbacks=[checkpointer], verbose=1,
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
 
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(test_tensors, test_targets, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.load_weights(file_path)
mushroom_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
test_accuracy = 100*np.sum(np.array(mushroom_predictions)==np.argmax(test_targets, axis=1))/len(mushroom_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

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