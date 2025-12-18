import os
import sys
import torch
import numpy as np
import torch
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, quaternion_to_matrix, matrix_to_rotation_6d



def npy_analyser(folder):
    """Analyse the npy files in a folder and return the different formats.
    
    Example:
    livox
    |--- 000000.npy
    |--- 000000_intensity.npy
    |--- 000001.npy
    |--- ...

    >>> npy_analyser("livox")
    {"", "intensity"}
        
    """
    formats = set()
    for file in filter(lambda f: f[-3:] == "npy", os.listdir(folder)):
        if "_" not in file:
            formats.add("")
            continue
        file_ext = file.split("_")[-1].split(".")[0]
        formats.add(file_ext)
    return formats


def select_sequence(traj, keys, start, length):
    return {
        key: traj[key][start : start + length] for key in keys
    }

def get_rel_quat(quat, reference_quat):
    """Return the relative rotations in quaternion between two sequences of quaternions"""
    inv_quat = quaternion_invert(reference_quat)
    return quaternion_multiply(quat, inv_quat)

def quat_to_6drot(quat):
    return matrix_to_rotation_6d(quaternion_to_matrix(quat))

def dict_to_tensor(tensor: torch.Tensor, flatten_temporal_dim):
    if flatten_temporal_dim:
        return torch.cat([value.flatten(start_dim=1) for value in tensor.values()], dim=-1)
    else:
        return torch.cat([value.flatten(start_dim=2) for value in tensor.values()], dim=-1)


def tensor_to_dict(tensor: torch.Tensor, keys, dict_config):
    list_len = [np.prod(size) for key, size in dict_config if key in keys]
    tensor_split = torch.spit(tensor, list_len, dim=-1)
    
    return {
        key: tensor for key, tensor in zip(keys, tensor_split)
    }

def dict_flatten(d, format_key=lambda _, sk:sk):
    """Take a tree of dict and flatten it.
    The key_format function is used to format the key of the subdicts.
    An example of the key_format function is: 
        format_key = lambda k, sk: f"{k}.{sk}"
    By default, the key_format function is the identity function with the subkey as the key.   
    
    Args:
        d (dict): A tree of dict
        format_key (callable): A function that takes two arguments, the key and the subkey, and returns the new key.
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat_dict.update({format_key(key, subkey): subvalue for subkey, subvalue in dict_flatten(value, format_key).items()})
        else:
            flat_dict[key] = value
    return flat_dict
