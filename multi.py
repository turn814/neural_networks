import numpy as np
import random
import cv2

def double(x, y):
    rand1 = random.randint(0, (len(x) - 1))
    rand2 = random.randint(0, (len(x) - 1))
    label = (y[rand1] * 10) + y[rand2]
    shift1 = np.roll(np.roll(x[rand1], random.randint(-3, 3), axis=0), random.randint(-3, 3), axis=1)
    shift2 = np.roll(np.roll(x[rand2], random.randint(-3, 3), axis=0), random.randint(-3, 3), axis=1)
    conc = np.hstack((shift1, shift2))
    resize = cv2.resize(conc, (28, 28))
    resize = np.array(resize)
    return resize, label
