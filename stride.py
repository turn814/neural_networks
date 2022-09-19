import mnist_classifier
from mnist_classifier import *

import tensorflow as tf
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


def stride(image, good_coords):
    lst = [image[:28, :28].tolist()]
    for i in range(len(image) - 28):
        for n in range(len(image) - 28):
            if i == 1 and n == 1:
                continue
            nxt = image[i:(28+i), n:(28+n)].tolist()
            lst.append(nxt)
    arr_labels = np.zeros(len(lst))
    for i in good_coords:
        arr_labels[(i[1] * (len(image) - 28) + i[0])] = 1
    arr_strided = np.reshape(lst, (len(lst), 28, 28))
    return arr_strided, arr_labels

def surround(image, size):
    length = len(image)
    horiz = np.random.randint(size - length)
    vert = np.random.randint(size - length)
    good_coords = [[int(horiz), int(vert)]]
    surrounded = np.pad(image, ((vert, size-length-vert) ,(horiz, size-length-horiz)), 'constant')
    return surrounded, good_coords

def quadsurround(arr_surrounded, lst_good_coords):
    surrounded_1 = np.concatenate((arr_surrounded[0], arr_surrounded[1]))
    surrounded_2 = np.concatenate((arr_surrounded[2], arr_surrounded[3]))
    surrounded = np.hstack((surrounded_1, surrounded_2))
    good_coords = np.reshape(np.concatenate(lst_good_coords), (-1, 2))
    return surrounded, good_coords


## pre-processing = striding and cropping
x_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_data.npy')
y_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_labels.npy')
x_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_data.npy')
y_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_labels.npy')
print('loaded mnist')

train_length = len(x_train)
test_length = len(x_test)
x, y = surround(x_train[0], 40)
x, y = stride(x, y)
affnist_x_train = [x]                     # list of array of images
affnist_y_train = [y]                     # list of array of labels
for i in range(499):
    surrounded, good_coords = surround(x_train[np.random.randint(60000)], 40)
    arr_strided, arr_labels = stride(surrounded, good_coords)
    affnist_x_train.append(arr_strided)
    affnist_y_train.append(arr_labels)
aff_x_train = np.reshape(affnist_x_train, (500 * 144, 28, 28))
aff_y_train = np.reshape(affnist_y_train, 500 * 144)
print('train data created')

x, y = surround(x_test[0], 40)
x, y = stride(x, y)
affnist_x_test = [x]
affnist_y_test = [y]
for i in range(99):
    surrounded, good_coords = surround(x_test[np.random.randint(10000)], 40)
    arr_strided, arr_labels = stride(surrounded ,good_coords)
    affnist_x_test.append(arr_strided)
    affnist_y_test.append(arr_labels)
aff_x_test = np.reshape(affnist_x_test, (100 * 144, 28, 28))
aff_y_test = np.reshape(affnist_y_test, 100 * 144)
print('test data created')

aff_x_train, aff_x_test = aff_x_train / 225.0, aff_x_test / 225.0
aff_x_train = aff_x_train[..., tf.newaxis].astype("float32")
aff_x_test = aff_x_test[..., tf.newaxis].astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices((aff_x_train, aff_y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((aff_x_test, aff_y_test)).batch(32)
print('pre-processed')


## binary digit identification model
binmodel = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                       tf.keras.layers.Dense(128, activation='relu'),
                                       tf.keras.layers.Dense(2)
                                       ])

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
print('pre-GradientTape')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = binmodel(images, training=True)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, binmodel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, binmodel.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = binmodel(images, training=False)
    t_loss = loss_obj(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
EPOCHS = 5
print("Binary Classifier Model:")
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result()}, '
          f'Accuracy: {train_accuracy.result() * 100}, '
          f'Test Loss: {test_loss.result()}, '
          f'Test Accuracy: {test_accuracy.result() * 100}')

# binmodel.save("binmodel.h5")
# print('saved')


rando = np.random.randint(10000)
x, y = surround(x_test[rando], 40)
x, y = stride(x, y)
combined_x_test = [x]
combined_y_test = [y]
labels = [[y_test[rando]] * 144]
for i in range(199):
    rando = np.random.randint(10000)
    surrounded, good_coords = surround(x_test[rando], 40)
    arr_strided, arr_labels = stride(surrounded, good_coords)
    combined_x_test.append(arr_strided)
    combined_y_test.append(arr_labels)
    labels.append([y_test[rando]] * 144)
combined_x_test = np.reshape(combined_x_test, (200 * 144, 28, 28))
combined_y_test = np.reshape(combined_y_test, 200 * 144)
labels = np.reshape(labels, 200 * 144)
print('combination test data created')

combined_x_test = combined_x_test / 225.0
combined_x_test = combined_x_test[..., tf.newaxis].astype("float32")
combined_test_ds = tf.data.Dataset.from_tensor_slices((combined_x_test, combined_y_test)).batch(32)
counter = 0
for x, y in combined_test_ds:
    predictions = binmodel.predict(x)
    classes = np.argmax(predictions, axis=1)
    for i in range(len(classes)):
        if classes[i] == 0:
            combined_x_test[(counter * 32) + i] = 0
    counter += 1
temp_test = np.zeros((len(combined_x_test), 28, 28, 1))
temp_labels = np.zeros(len(combined_x_test))
counter = 0
count = 0
for i in range(len(combined_x_test)):
    if isinstance(combined_x_test[i], int):
        counter += 1
        continue
    temp_test[count] = combined_x_test[i]
    temp_labels[count] = labels[i]
    count += 1
combined_x_test = temp_test[:(len(combined_x_test) - counter)]
labels = temp_labels[:(len(combined_x_test) - counter)]
combined_test_ds = tf.data.Dataset.from_tensor_slices((combined_x_test, labels)).batch(32)
print('pre-processed')

counter = 0
for x, y in combined_test_ds:
    predictions = binmodel.predict(x)
    classes = np.argmax(predictions, axis=1)
    for i in range(len(classes)):
        if classes[i] == y[i]:
            counter += 1
print('Combined \n'
      f'Test Accuracy: {counter / len(combined_x_test * 100)}')

## initializer quad_image strided
rando = np.random.randint(10000)
surrounded, good_coords = surround(x_test[rando], 40)
arr_surrounded = [surrounded]
lst_good_coords = [good_coords]
lst_catlabels = [[y_test[rando]] * int(((80 - 28)/2)**2)]
for i in range(3):
    rando = np.random.randint(10000)
    surrounded, good_coords = surround(x_test[rando], 40)
    arr_surrounded.append(surrounded)
    lst_good_coords.append(good_coords)
    lst_catlabels.append([y_train[rando]] * int(((80 - 28)/2)**2))
surrounded, good_coords = quadsurround(arr_surrounded, lst_good_coords)
arr_strided, arr_labels = stride(surrounded, good_coords)
arr_strided = [arr_strided]
lst_binlabels = [arr_labels]
lst_catlabels = [lst_catlabels]

## dataset of quad_images
for i in range(49):
    rando = np.random.randint(10000)
    surrounded, good_coords = surround(x_test[rando], 40)
    arr_surrounded = [surrounded]
    lst_good_coords = [good_coords]
    temp_lst_cat_labels = [[y_test[rando]] * int(((80 - 28)/2)**2)]
    for i in range(3):
        rando = np.random.randint(10000)
        surrounded, good_coords = surround(x_test[rando], 40)
        arr_surrounded.append(surrounded)
        lst_good_coords.append(good_coords)
        temp_lst_cat_labels.append([y_test[rando]] * int(((80 - 28)/2)**2))
    surrounded, good_coords = quadsurround(arr_surrounded, lst_good_coords)
    temp_arr_strided, temp_arr_labels = stride(surrounded, good_coords)
    arr_strided.append(temp_arr_strided)
    lst_binlabels.append(temp_arr_labels)
    lst_catlabels.append(temp_lst_cat_labels)
arr_strided = np.reshape(arr_strided, (-1, 28, 28))
lst_binlabels = np.reshape(lst_binlabels, -1)
lst_catlabels = np.reshape(lst_catlabels, -1)
print('quad data created')

arr_strided = arr_strided / 225.0
arr_strided = arr_strided[..., tf.newaxis].astype("float32")
arr_strided_ds = tf.data.Dataset.from_tensor_slices((arr_strided, lst_binlabels)).batch(32)
counter = 0
for x, y in arr_strided_ds:
    predictions = binmodel.predict(x)
    classes = np.argmax(predictions, axis=1)
    for i in range(len(classes)):
        if classes[i] == 0:
            arr_strided[(counter * 32) + i] = 0
    counter += 1
temp_test = np.zeros((len(arr_strided), 28, 28, 1))
temp_labels = np.zeros(len(arr_strided))

top = 0
for i in lst_binlabels:
    if i == 0:
        top += 1
bot = 0
for i in lst_binlabels:
    if i == 1:
        bot += 1
print(f'Binary Model 0 Predictions: {counter}, '
      f'Binary Model 0 Labels: {top}'
      f'Binary Model 1 Labels: {bot}')

counter = 0
count = 0
for i in range(len(arr_strided)):
    if isinstance(arr_strided[i], int):
        counter += 1
        continue
    temp_test[count] = arr_strided[i]
    temp_labels[count] = lst_catlabels[i]
    count += 1
arr_strided = temp_test[:(len(arr_strided) - counter)]
lst_cat_labels = temp_labels[:(len(arr_strided) - counter)]
quad_test_ds = tf.data.Dataset.from_tensor_slices((arr_strided, lst_catlabels)).batch(32)
print('pre_processed')

counter = 0
for x, y in quad_test_ds:
    predictions = model.predict(x)
    classes = np.argmax(predictions, axis=1)
    for i in range(len(classes)):
        if classes[i] == y[i]:
            counter += 1

print('Quad: \n'
      f'Test Accuracy: {counter / len(arr_strided) * 100}')
