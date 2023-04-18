import hashlib
import os
import pickle
from typing import Protocol

import numpy as np


def hash_args(*args)->str:
    arg_hash = hashlib.new("sha256")
    for arg in args:
        if type(arg) is type:
            arg_hash.update(str(arg).encode())
        if type(arg) is np.ndarray:
            arg_hash.update(str(np.sum(arg)).encode())
        else:
            arg_hash.update(str(arg).encode())

    return arg_hash.hexdigest()

class Serializable(Protocol):
    def serialize(self, file_name):
        if not os.path.exists(".out/"):
            os.mkdir(".out/")

        if not os.path.exists(".out/persistent_experiments/"):
            os.mkdir(".out/persistent_experiments/")

        file_path = '.out/persistent_experiments/' + file_name

        if os.path.exists(file_path):
            raise Exception(f"Cannot serialized value {file_path}. It already exists.")

        fp=open(file_path,'wb')
        pickle.dump(self.to_serialized(), fp)
        fp.close()

    def to_serialized(self) -> dict:
        pass

    def from_serialized(self, serialized):
        pass

    def deserialize(self, file_name):
        if not os.path.exists(".out/"):
            os.mkdir(".out/")

        if not os.path.exists(".out/persistent_experiments/"):
            os.mkdir(".out/persistent_experiments/")

        file_path = '.out/persistent_experiments/' + file_name

        if not os.path.exists(file_path):
            raise Exception("Could not deserialize, didn't exist")

        fp=open(file_path,'rb')
        serialized_value = pickle.load(fp)
        fp.close()
        return self.from_serialized(serialized_value)

    def check_for_serialized_version(self, *args):
        """Returns whether the result has been serialized, along with it's serial hash"""
        file_name = hash_args(args) + ".gdrun"

        if not os.path.exists(".out/"):
            os.mkdir(".out/")

        if not os.path.exists(".out/persistent_experiments/"):
            os.mkdir(".out/persistent_experiments/")

        file_path = '.out/persistent_experiments/' + file_name
        if not os.path.exists(file_path):
            return False, file_name
        print("------Experiment serializer-------")
        print("A gradient descent run was found to have been performed before!")
        print("inputs were")
        print("")
        print(args)
        print("")
        print("If you believe this to be false, delete the file ", file_path)
        print("----------------------------------")
        return True, file_name