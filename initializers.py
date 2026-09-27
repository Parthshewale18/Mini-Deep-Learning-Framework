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

def initialize(fan_in, fan_out, method="glorot"):
        if method == "glorot":
              return glorot_uniform(fan_in, fan_out)
        elif method == "he":
              return he_normal(fan_in)
        elif method == "lecun":
              return lecun_normal(fan_in)
        else:
             raise ValueError(f"Unknown init: {method}")