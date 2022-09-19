# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

import tensorflow as tf
import numpy as np

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
        arr_labels[i[0] * (len(image) - 28) + i[1]] = 1
    arr_strided = np.reshape(lst, (len(lst), 28, 28))
    return arr_strided, arr_labels

def surround(image):
    length = len(image)
    horiz = np.random.randint(40 - length)
    vert = np.random.randint(40 - length)
    good_coords = [(vert, horiz)]
    for i in range(5):
        for n in range(5):
            v_coord = vert + 2 - i
            h_coord = horiz + 2 - n
            if i == 2 and n == 2:
                continue
            if v_coord < 0 or v_coord >= 40 - length or h_coord < 0 or h_coord >= 40 - length:
                continue
            good_coords.append((v_coord, h_coord))
    surrounded = np.pad(image, ((vert, 40-length-vert) ,(horiz, 40-length-horiz)), 'constant')
    return surrounded, good_coords


# x_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_data.npy')
# y_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_labels.npy')
# x_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_data.npy')
# y_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_labels.npy')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('loaded mnist')


binmodel = tf.keras.models.load_model('./binmodel.h5')
ogmodel = tf.keras.models.load_model('./ogmodel.h5')
print('models loaded')

rando = np.random.randint(10000)
x, y = surround(x_test[rando])
x, y = stride(x, y)
combined_x_test = [x]
combined_y_test = [y]
labels = [[y_test[rando]] * 144]
for i in range(199):
    rando = np.random.randint(10000)
    surrounded, good_coords = surround(x_test[rando])
    arr_strided, arr_labels = stride(surrounded, good_coords)
    combined_x_test.append(arr_strided)
    combined_y_test.append(arr_labels)
    labels.append([y_test[rando]] * 144)
combined_x_test = np.reshape(combined_x_test, (200 * 144, 28, 28))
combined_y_test = np.reshape(combined_y_test, 200 * 144)
labels = np.reshape(labels, 200 * 144)
print('combination test data created')

combined_x_test = combined_x_test / 225.0
x_test = x_test[..., tf.newaxis].astype("float32")
print('pre-processed')

for i in range(200 * 144):
    if binmodel(combined_x_test[i], training=False) == 0:
        np.delete(combined_x_test, i)
        np.delete(combined_y_test, i)
        np.delete(labels, i)

counter = 0
for i in range(len(combined_x_test)):
    if ogmodel(combined_x_test[i], training=False) == labels[i]:
        counter += 1

print(counter)
