from typing import List, Dict, Tuple
import numpy as np
import torch
import os
import yaml
from src.utils.timestamps import get_end_of_time
from src.loader import str_to_loader, loads_timestamps, load_profile
from src.utils.files import get_files
from ..core import AbstractDataset, AbstractLoader

class TartanKittiDataset(AbstractDataset):
    r""" A :class:`AbstractRobotDataset` subclass that manage Kitti format from Tartan drive

    This will never be used like this but be applied to wrapper,
    that will convert them into dataset object from libraries.

    Args:
        directory (str) :
            The directory where the dataset is stored
        keys (List) :
            The keys of the dataset
        dataset_profile (str) :
            The profile of the dataset

    Remarks :
         * The object that can be quite big (not the one that be loaded) need
           to be stocked in numpy style array (panda, tensor, ...)
           this is a TODO: timestamps (dict) -> timestamps (np.arrays) (pointing manager)
           For more details : https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    """
    synchronous = False
    _keys : np.ndarray
    profile : Dict[str, str]
    files : Dict[str, str]
    loaders : Dict[str, AbstractLoader]
    timestamps : Dict[str, np.ndarray]
    timeline : List[Tuple[str, int]]
    
    def __init__(self, 
                 directory, 
                 keys,
                 dataset_profile="tartan_kitti.yaml",
                 *args, **kwargs
                 ):
        
        # Init variables
        self.profile = load_profile(dataset_profile)
        self.files = get_files(directory)
        self.keys = keys  

        # Check key integrity
        if not set(keys).issubset(self.files.keys()):
            raise KeyError(f"keys {keys} not in the dataset")     

    @property
    def profile(self):
        return self._profile
    
    @profile.setter
    def profile(self, profile: dict):
        self._profile = profile

    @property
    def keys(self):
        return self._keys
    
    @keys.setter
    def keys(self, keys: list):
        """
        Even if keys will not be big, it it important to work 
        with similare objects
        """
        super(TartanKittiDataset, type(self)).keys.__set__(self, keys)
        
        if not set(keys).issubset(self.files.keys()):
            raise KeyError(f"keys {keys} not in the dataset")
        
        self.init()

    @property
    def shape(self):
        return {key: self.loaders[key].shape for key in self.keys}

    def init(self):
        if len(self.keys) == 0:
            return
        self.init_loaders()
        self.init_timelines()

    def init_loaders(self):
        self.loaders = {
            key: str_to_loader[self.profile[key]](self.files[key]) for key in self.keys
            }
        self.timestamps = loads_timestamps(self.keys, self.files)
        self.end_of_time = get_end_of_time(self.timestamps) + 1.0

    def init_timelines(self):
        _indexes = np.zeros(len(self.keys), dtype=int)
        _timelines = np.array([self.timestamps[key][0] for key in self.keys])
        self.timeline = [] #[(key, index)]

        # Keep until __next__ is changed : will be deleted
        self.indexes = _indexes
        self.timelines = _timelines
        
        while True:
            # Here timeline_idx is the index in the full_keys list
            timeline_idx = np.argmin(_timelines)
            timeline_key = self.keys[timeline_idx]

            if _timelines[timeline_idx] >= self.end_of_time:
                break
            
            _indexes[timeline_idx] += 1

            if _indexes[timeline_idx] == len(self.timestamps[timeline_key]):
                _timelines[timeline_idx] = self.end_of_time
                return (timeline_key, _indexes[timeline_idx]-1)
            
            _timelines[timeline_idx] = self.timestamps[timeline_key][_indexes[timeline_idx]]
            self.timeline.append((timeline_key, _indexes[timeline_idx]-1))   # Return the key and the index of the data


    def __getitem__(self, idx):
        key, index = self.timeline[idx]
        return torch.tensor({
            "key": key,
            "data": self.loaders[key][index],
            "timestamp": self.timestamps[key][index]
        })


    # def __iter__(self):
    #     self.init_timelines()
    #     return self
    
    # def __next__(self):
    #     """ Return the next batch of data according to the timestamps.
    #     It uses the indexes to keep track of the last data loaded.
    #     A timeline that is over will be set to the end of the time.
        
    #     """
    #     # Here timeline_idx is the index in the full_keys list
    #     timeline_idx = np.argmin(self.timelines)
    #     timeline_key = self.keys[timeline_idx]

    #     if self.timelines[timeline_idx] >= self.end_of_time:
    #         raise StopIteration
        
    #     self.indexes[timeline_idx] += 1

    #     if self.indexes[timeline_idx] == len(self.timestamps[timeline_key]):
    #         self.timelines[timeline_idx] = self.end_of_time
    #         return (timeline_key, self.indexes[timeline_idx]-1)
        
    #     self.timelines[timeline_idx] = self.timestamps[timeline_key][self.indexes[timeline_idx]]
    #     return (timeline_key, self.indexes[timeline_idx]-1)   # Return the key and the index of the data

