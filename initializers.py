""" Initializers for Neural Networks"""

import math
import numpy as np

def glorot_uniform(fan_in, fan_out):
    limit = math.sqrt(6 / (fan_in + fan_out))

    return np.random.uniform(-limit, limit)

def he_normal(fan_in):
    std = math.sqrt(2 / fan_in)

    return np.random.normal(0, std)

def lecun_normal(fan_in):
    std = math.sqrt(1 / fan_in)

    return np.random.normal(0, std)