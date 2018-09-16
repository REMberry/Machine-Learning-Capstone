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

import itertools
import numpy as np
import os
from keras.utils import np_utils
from tqdm import tqdm
from keras.preprocessing import image      
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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

file_path = 'saved_models/weights.best.from_V3-11.hdf5'
batch_size = 32

test_files, test_targets = load_dataset('data/test')

test_tensors = paths_to_tensor(test_files).astype('float32')/255

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
model.add(Dense(10, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.load_weights(file_path)


test_generator = ImageDataGenerator(rescale=1./255)

test_data_generator = test_generator.flow_from_directory(
    'data/test', # Put your path here
     target_size=(150, 150),
     batch_size=batch_size,
     shuffle=False)

test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)

#predictions = model.predict_generator
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())   

report = classification_report(true_classes, predicted_classes) #target_names=class_labels)
print(report)   

cm = confusion_matrix(y_true=true_classes, y_pred=predicted_classes)

labels = ['001','002','003','004',
            '005','006','007','008','009','010']


plt.figure()
plot_confusion_matrix(cm, classes=labels,
                      title='Confusion matrix, without normalization')
plt.show()
#print(cm)
#pl.matshow(cm)
#pl.title('Confusion matrix of the classifier')
#pl.colorbar()
#pl.show()




#print(cm)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)
#plt.title('Confusion matrix of the classifier')
#fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()
#mushroom_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
#test_accuracy = 100*np.sum(np.array(mushroom_predictions)==np.argmax(test_targets, axis=1))/len(mushroom_predictions)
#print('Test accuracy: %.4f%%' % test_accuracy)