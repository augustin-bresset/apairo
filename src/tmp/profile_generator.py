import os
import sys

import yaml


from src.utils.utils import npy_analyser

main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, main_dir)
example_dir = os.path.join(main_dir, "data/tartan_2_kitti/example1")

def write_dict_to_yaml(dict, path):
    with open(path, 'w') as file:
        documents = yaml.dump(dict, file)

def type_files(file1, file2):
    """Return the basename of the file.
    Example:
    >>> type_file('042_depth.pny', '042_rgb.pny') 
    (_depth, '_rgb')

    So it detects the basename by comparing the files.
    """
    format1, format2 = "", ""
    for i, (c1, c2) in enumerate(zip(file1, file2)):
        if c1 != c2:
            format1 = file1[i:]
            format2 = file2[i:]
            break
    return format1, format2


def generate_folder_profile():
    profile = {}
    for dir in os.listdir(example_dir):
        folder_path = os.path.join(example_dir, dir)
        if os.path.isdir(folder_path):
            for file_ in os.listdir(folder_path):
                file, ext = file_.split(".")
                if ext in {"png", "jpg"}:
                    if profile.get(dir) in {"npy", "npys"}:
                        profile[dir] = "npys_img"
                        break
                    profile[dir] = "img"

                if ext == "npy":
                    if profile.get(dir) == "img":
                        profile[dir] = "npys_img"
                        break
                    if profile.get(dir) == "npy":
                        profile[dir] = "npys"
                    else:
                        profile[dir] = "npy"
                                    
    return profile


if __name__ == "__main__":
    profile = generate_folder_profile()
    write_dict_to_yaml(profile, os.path.join(main_dir, "profile.yaml"))
    print("Profile generated")
