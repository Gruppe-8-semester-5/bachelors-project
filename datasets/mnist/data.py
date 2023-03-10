import numpy as np
from datasets.mnist.digit import Digit

def data_to_digits(datalist: list) -> list[Digit]: 
    """Converts a raw list into a list of digit objects"""
    return list(map(lambda data: Digit(data), datalist))

def data_to_array(datalist: list) -> np.ndarray: 
    """Converts a raw list into a vector of labels and matrix of features"""
    # TODO: Optmimize if need be
    digits = data_to_digits(datalist)
    label_map = map(lambda digit: digit.get_label(), digits)
    feature_map = map(lambda digit: digit.get_features(), digits)
    return  np.array(list(feature_map)), np.array(list(label_map))
