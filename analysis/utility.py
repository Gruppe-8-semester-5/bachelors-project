import numpy as np
import os
import os.path


def euclid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The euclidean distance. a and b should be vectors, or points in n-dim space"""
    sum_sq = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sum_sq += diff**2
    return np.sqrt(sum_sq)

def dump_array_to_csv(array: np.ndarray, file_name: str, override = False, prepend=""):
    if not file_name.endswith(".csv"):
        raise Exception("Please make sure the filename ends with .csv !")
    target_file = f"./.out/{prepend}_{file_name}"
    if not os.path.exists(".out/"):
        os.mkdir(".out/")
    if os.path.isfile(target_file):
        if override:
            os.remove(target_file)
        else:
            raise Exception("Could not dump to file. It already exists. Use override = True if you want to override it")
    np.savetxt(target_file, array, delimiter=",")