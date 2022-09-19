import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

print("TensorFlow version: ", tf.__version__)

##local mnist load attempt
##mnist = mnist.load_data(path='/mnt/net/i2x256-ai01/data/mnist')

# mnist = open('/mnt/net/i2x256-ai01/data/mnist', 'r')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_data.npy')
# y_train = np.load('/mnt/net/i2x256-ai01/data/mnist/train_labels.npy')

# x_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_data.npy')
# y_test = np.load('/mnt/net/i2x256-ai01/data/mnist/val_labels.npy')

x_train, x_test = x_train / 225.0, x_test / 225.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='Adam', loss=loss, metrics='Accuracy')
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
