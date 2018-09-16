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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

top_model_weights_path = 'saved_models/bottleneck_fc_model.h5'
top_model_fine_weights_path = 'saved_models/bottleneck_fc_model_fine.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
nb_train_samples = 976
nb_validation_samples = 100
batch_size = 32

def save_bottlebeck_features(train_tensors, valid_tensors, test_tensors):

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('validation data prediction')
    
    bottleneck_features_valid = model.predict(valid_tensors)
    np.save('bottleneck_features_valid.npy', bottleneck_features_valid)

    bottleneck_features_test = model.predict(test_tensors)
    np.save('bottleneck_features_test.npy', bottleneck_features_test)

    bottleneck_features_train = model.predict(train_tensors)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)



def train_top_model(train_targets, valid_targets, test_targets, test_tensors):

    input_tensor = Input(shape=(150,150,3))
    base_model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)

    print('Model loaded.')

    top_model = Sequential()
    top_model.add(BatchNormalization(input_shape=base_model.output_shape[1:]))
    top_model.add(GlobalAveragePooling2D())
    top_model.add(Dense(32, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(10, activation='softmax'))

    top_model.load_weights(top_model_weights_path)

    model = Model(input= base_model.input, output= top_model(base_model.output))

    for layer in model.layers[:25]:
        layer.trainable = False


    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

    model.summary()

    checkpointer = ModelCheckpoint(filepath=top_model_fine_weights_path, 
                               verbose=1, save_best_only=True)

    train_datagen = ImageDataGenerator(
        rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.2,
        shear_range=0.15,  # set range for random shear
        zoom_range=0.15,  # set range for random zoom
        channel_shift_range=0.0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,
        rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,)

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=10,
        validation_data=validation_generator,
        verbose = 1,
        shuffle = True,
        callbacks=[checkpointer],
        validation_steps=nb_validation_samples // batch_size)

#    history = model.fit(train_data, train_targets,
#              #steps_per_epoch = 100,
#              epochs=200, #300
#              batch_size=32,
#              shuffle = True,
 #             verbose=1,
#              validation_data=(valid_data, valid_targets),
#              callbacks=[checkpointer])
    
    #model.save_weights(top_model_fine_weights_path)
    model.load_weights(top_model_fine_weights_path)

    pred_values = [(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    mushroom_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    true_predictions = np.argmax(test_targets, axis=1)

    test_accuracy = 100*np.sum(np.array(mushroom_predictions)==np.argmax(test_targets, axis=1))/len(mushroom_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    cm = confusion_matrix(y_true=true_predictions, y_pred=mushroom_predictions)
    labels = ['001','002','003','004',
                '005','006','007','008','009','010']
    plt.figure()
    plot_confusion_matrix(cm, classes=labels,
                        title='Confusion matrix, without normalization')
    plt.show()

    report = classification_report(true_predictions ,  mushroom_predictions)
    print(report)   




train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

#save_bottlebeck_features(train_tensors, valid_tensors, test_tensors)

print('train top model')

train_top_model(train_targets, valid_targets, test_targets, test_tensors)