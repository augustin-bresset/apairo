import os


def empty_dir(directory, remove=False):
    """Empty a directory and optionally remove it.
    
    It removes all the files and directories inside the directory.
    """
    for f in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, f)):
            empty_dir(os.path.join(directory, f), True)
        else:
            os.remove(os.path.join(directory, f))
    if remove:
        os.rmdir(directory) 