from apairo.core.profiled_dataset import ProfiledDataset


class SemanticKittiDataset(ProfiledDataset):
    """SemanticKITTI dataset -- driving LiDAR with dense semantic labels.

    Keys: ``lidar`` (float32, shape (N, 4)), ``labels`` (int64, lower 16 bits = semantic class).

    Example::

        ds = SemanticKittiDataset("/data/kitti/dataset", keys=["lidar", "labels"])
        sample = ds[0]
        # sample.data["lidar"]  -> np.ndarray (N, 4)
        # sample.data["labels"] -> np.ndarray (N,)
    """

    _profile = "semantic_kitti.yaml"
