import os
from typing import List, Dict
import numpy as np
import torch
from test.paths import tmp_path


def create_timestamps_file(directory, start, end, n, freq_var=0.0, overwrite=False):
    """Create a timestamps file.

    The file is a numpy array with the timestamps of the data.
    His name is always `timestamps.txt`.

    To create a timestamps file that simulates a real timeline, the timestamps are created with a normal distribution
    with a mean of `end` - `start` / n.
    """
    if os.path.isfile(os.path.join(tmp_path, directory, "timestamps.txt")) and not overwrite:
        return
    mean = (end - start) / n

    timestamps = np.random.normal(mean, freq_var, n)
    timestamps = np.cumsum(timestamps)
    if not os.path.isdir(os.path.join(tmp_path, directory)):
        os.makedirs(os.path.join(tmp_path, directory), exist_ok=True)
    if tmp_path not in directory:
        directory = os.path.join(tmp_path, directory)
    file_path = os.path.join(directory, "timestamps.txt")
    with open(file_path, "w") as f:
        for t in timestamps:
            f.write(f"{t}\n")


def create_random_pt_file(keys: List, len_: int, shapes: List, filename="data.pt", directory="", dt=0.1):
    """Generate a random .pt file with the given shape and length.

    A pt file is a dictionary with keys and :class:`torch.Tensor` values.

    Args:
        keys (list): List of keys for the dictionary.
        len_ (int): Length of the first dimension of the data.
        shapes (list): List of shapes for the tensors for each key.
        filename (str): Name of the file.
        directory (str): Directory where to save the file.
        overwrite (bool): If True, overwrite existing files.
    """
    data: Dict[str, torch.Tensor] = {}
    for key, shape in zip(keys, shapes):
        if isinstance(shape, int):
            data[key] = torch.tensor(np.random.rand(len_, shape))
        else:
            data[key] = torch.tensor(np.random.rand(len_, *shape))
    dt = np.cumsum([dt] * len_)
    data["dt"] = torch.from_numpy(dt)
    if not os.path.isdir(os.path.join(tmp_path, directory)):
        os.makedirs(os.path.join(tmp_path, directory), exist_ok=True)
    file_path = os.path.join(tmp_path, directory, filename)
    torch.save(data, file_path)


def create_random_pt_files(n_files, len_, shape, file_spec="", directory="", overwrite=False):
    """Generate n_files random .pt files with the same shape and length.

    The name of the files will be their index on 6 digits, e.g. 000001.pt.
    If :attr:`file_spec` is not empty, it will be added before the index, e.g. 000001_{file_spec}.pt.

    Args:
        n_files (int): Number of files to generate.
        len_ (int): Length of the first dimension of the data.
        shape (tuple): Shape of the data.
        file_spec (str): Optional specification to add to the file name.
        directory (str): Directory where to save the files.
        overwrite (bool): If True, overwrite existing files.
    """
    # directory_path = os.path.join(tmp_path, directory)
    for i in range(n_files):
        if file_spec:
            create_random_pt_file(len_, shape, f"{i:06}_{file_spec}.pt", directory, overwrite)
        else:
            create_random_pt_file(len_, shape, f"{i:06}.pt", directory, overwrite)


def create_npy_file(data, filename, directory="", overwrite=False):
    if tmp_path not in directory:
        directory = os.path.join(tmp_path, directory)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    elif os.path.isfile(os.path.join(directory, filename)) and not overwrite:
        return
        raise FileExistsError(f"File {filename} already exists in {directory}")

    file_path = os.path.join(directory, filename)
    np.save(file_path, data)


def create_random_npy_file(len_, shape, filename, directory="", overwrite=False):
    if isinstance(shape, int):
        data = np.random.rand(len_, shape)
    else:
        data = np.random.rand(len_, *shape)
    create_npy_file(data, filename, directory, overwrite)


def create_random_npy_files(n_files, shape, directory="", file_spec="", overwrite=False):
    """Generate n_files random .npy files with the same shape and length.

    The name of the files will be their index on 6 digits, e.g. 000001.npy.
    If :attr:`file_spec` is not empty, it will be added before the index, e.g. 000001_{file_spec}.npy.

    Args:
        n_files (int): Number of files to generate.
        shape (int or tuple): Shape of the data.
        file_spec (str): Optional specification to add to the file name.
        directory (str): Directory where to save the files.
        overwrite (bool): If True, overwrite existing files.
    """
    for i in range(n_files):
        data = np.random.rand(*shape)
        if file_spec:
            filename = f"{i:06}_{file_spec}.npy"
        else:
            filename = f"{i:06}.npy"
        create_npy_file(data, filename, directory, overwrite)


def create_random_images(n_images, shape, directory="", overwrite=False):
    from PIL import Image

    if tmp_path not in directory:
        directory = os.path.join(tmp_path, directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(n_images):
        image = np.random.randint(0, 255, (*shape, 3), dtype=np.uint8)
        file_path = os.path.join(directory, f"{i:06d}.png")
        Image.fromarray(image, "RGB").save(file_path)
