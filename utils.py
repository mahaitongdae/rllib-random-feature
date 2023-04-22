import numpy as np


def angle_normalize(th):
    return ((th + np.pi) % (2 * np.pi)) - np.pi