import json
import os
import pickle

def serialize(value, *args):
    file_name_hash_input = ""
    for arg in args:
        file_name_hash_input += str(arg)

    file_name = str(hash(file_name_hash_input)) + ".txt"

    if not os.path.exists(".out/"):
        os.mkdir(".out/")

    if not os.path.exists(".out/persistent_experiments/"):
        os.mkdir(".out/persistent_experiments/")

    file_path = '.out/persistent_experiments/' + file_name

    if os.path.exists(file_path):
        raise Exception(f"Cannot serialized value {file_path}. It already exists.")

    fp=open(file_path,'wb')
    pickle.dump(value, fp)
    fp.close()

def try_deserialize(*args):
    file_name_hash_input = ""
    for arg in args:
        file_name_hash_input += str(arg)

    file_name = str(hash(file_name_hash_input)) + ".txt"

    if not os.path.exists(".out/"):
        os.mkdir(".out/")

    if not os.path.exists(".out/persistent_experiments/"):
        os.mkdir(".out/persistent_experiments/")

    file_path = '.out/persistent_experiments/' + file_name

    if not os.path.exists(file_path):
        return False, None

    fp=open(file_path,'r')
    deserialized_value = pickle.load(fp)
    fp.close()
    return True, deserialized_value

def remove_serialized_value(*args):
    file_name_hash_input = ""
    for arg in args:
        file_name_hash_input += str(arg)

    file_name = str(hash(file_name_hash_input)) + ".txt"

    if not os.path.exists(".out/"):
        os.mkdir(".out/")

    if not os.path.exists(".out/persistent_experiments/"):
        os.mkdir(".out/persistent_experiments/")

    file_path = '.out/persistent_experiments/' + file_name

    if os.path.exists(file_path):
        os.remove(file_path)