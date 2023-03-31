import os
import pickle
from typing import Protocol

import numpy as np


class Serializable(Protocol):
    def serialize(self, hash_input):
        file_name = str(hash(hash_input)) + ".json"

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

    def deserialize(self, hash_input):
        file_name = str(hash(hash_input)) + ".json"

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

    def check_for_serialization(self, *args):
        """Returns whether the result has been serialized, along with it's serial hash"""
        print("--------------")
        print("Checking for serialization", args)
        print("--------------")
        file_name_hash_input = 0
        for arg in args:
            if type(arg) is np.ndarray:
                file_name_hash_input += hash(np.sum(arg))
            else:
                file_name_hash_input += hash(arg)
            print("-------TO -------")
            print(file_name_hash_input)
            

        file_name = str(hash(file_name_hash_input)) + ".json"
        print(file_name)

        if not os.path.exists(".out/"):
            os.mkdir(".out/")

        if not os.path.exists(".out/persistent_experiments/"):
            os.mkdir(".out/persistent_experiments/")

        file_path = '.out/persistent_experiments/' + file_name

        if not os.path.exists(file_path):
            return False, file_name_hash_input
        return True, file_name_hash_input