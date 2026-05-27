from apairo.core.profiled_dataset import ProfiledDataset


class Rellis3DDataset(ProfiledDataset):
    """RELLIS-3D dataset — off-road LiDAR with semantic labels.

    Keys: ``lidar`` (float32, shape (N, 4)), ``labels`` (int64).

    Example::

        ds = Rellis3DDataset("/data/RELLIS", keys=["lidar", "labels"])
        sample = ds[0]
        # sample.data["lidar"]  → np.ndarray (N, 4)
        # sample.data["labels"] → np.ndarray (N,)
    """

    _profile = "rellis.yaml"
