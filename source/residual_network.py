import numpy as np
from keras.utils import np_utils
from sklearn.datasets import load_files
from glob import glob
import os
from keras.preprocessing import image                  
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return ResNet50_model.predict(img)
    #return np.argmax(ResNet50_model.predict(img))

def mushroom_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    #print('Predicted:', decode_predictions(prediction))
    max_prediction = np.argmax(prediction)
    return max_prediction
    #return (max_prediction == 948)

def load_dataset(path):
    data = load_files(path)
    mushroom_files = np.array(data['filenames'])
    mushroom_targets = np_utils.to_categorical(np.array(data['target']), 10)
    return mushroom_files, mushroom_targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

dirname = os.path.dirname(__file__)
validpath = os.path.join(dirname, 'relative/path/to/file/you/want')

train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

mushroom_species = [item[20:-1] for item in sorted(glob("data/train/*/"))]

print('There are %d mushroom species.' % len(mushroom_species))
print('There are %s total mushroom images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training mushroom images.' % len(train_files))
print('There are %d validation mushroom images.' % len(valid_files))
print('There are %d test mushroom images.'% len(test_files))



# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

mushroom_files_short = train_files
mushroom_files_short = np.append(mushroom_files_short , valid_files)
mushroom_files_short = np.append(mushroom_files_short , test_files)
#mushroom_files_short.extend(valid_files)
#mushroom_files_short.extend(test_files)

lst = np.zeros(1000)

for mf in mushroom_files_short:
    res = mushroom_detector(mf)
    lst[res] = lst[res]+1
    print(res)

idx = (-lst).argsort()[:10]
print(idx)
print('1st most types are %d values as %d' % (lst[idx[0]] , idx[0]))
print('2nd most types are %d values as %d' % (lst[idx[1]] , idx[1]))
print('3rd most types are %d values as %d' % (lst[idx[2]] , idx[2]))
print('4th most types are %d values as %d' % (lst[idx[3]] , idx[3]))
print('5th most types are %d values as %d' % (lst[idx[4]] , idx[4]))

print('6th most types are %d values as %d' % (lst[idx[5]] , idx[5]))
print('7th most types are %d values as %d' % (lst[idx[6]] , idx[6]))
print('8th most types are %d values as %d' % (lst[idx[7]] , idx[7]))
print('9th most types are %d values as %d' % (lst[idx[8]] , idx[8]))
print('10th most types are %d values as %d' % (lst[idx[9]] , idx[9]))
#mushroom_detect = [mf for mf in mushroom_files_short if mushroom_detector(mf)]
#mushroom_rate = len(mushroom_detect)
#print("The Mushroom rate rate is: {}%".format(mushroom_rate))
