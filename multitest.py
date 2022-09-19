import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import numpy as np
import tensorflow as tf
import multi
import matplotlib
import matplotlib.pyplot as plt

# x_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_data.npy')
# y_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_labels.npy')
# x_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_data.npy')
# y_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_labels.npy')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('loaded')

new_x_train, new_y_train = multi.double(x_train, y_train)
new_x_test, new_y_test = multi.double(x_test, y_test)

new_x_train = [new_x_train.tolist()]
new_y_train = [new_y_train]
new_x_test = [new_x_test.tolist()]
new_y_test = [new_y_test]
for i in range(len(x_train) - 1):
    x_tr, y_tr = multi.double(x_train, y_train)
    x_tr = x_tr.tolist()
    new_x_train.append(x_tr)
    new_y_train.append(y_tr)
for i in range(len(x_test) - 1):
    x_te, y_te = multi.double(x_test, y_test)
    x_te = x_te.tolist()
    new_x_test.append(x_te)
    new_y_test.append(y_te)

new_x_train = np.reshape(new_x_train, newshape=(len(x_train), 28, 28))
new_y_train = np.reshape(new_y_train, newshape=len(x_train))
new_x_test = np.reshape(new_x_test, newshape=(len(x_test), 28, 28))
new_y_test = np.reshape(new_y_test, newshape=len(x_test))
print('doubled')

new_x_train, new_x_test = new_x_train / 225.0, new_x_test / 225.0
new_x_train = new_x_train[..., tf.newaxis].astype("float32")
new_x_test = new_x_test[..., tf.newaxis].astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((new_x_test, new_y_test)).batch(32)
print('pre-processed')

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(100)
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
        predictions = model(images, training=True)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_obj(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

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
