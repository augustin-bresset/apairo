from apairo.core.profiled_dataset import ProfiledDataset


class Rellis3DDataset(ProfiledDataset):
    """RELLIS-3D dataset -- off-road LiDAR with semantic labels.

    Keys: ``lidar`` (float32, shape (N, 4)), ``labels`` (int64).
    Optional: ``poses`` (float64, shape (3, 4)) -- one 3x4 pose matrix per frame,
    loaded from a per-sequence ``poses.txt`` file (one row of 12 floats per frame).

    Example::

        ds = Rellis3DDataset("/data/RELLIS", keys=["lidar", "labels"])
        sample = ds[0]
        # sample.data["lidar"]  -> np.ndarray (N, 4)
        # sample.data["labels"] -> np.ndarray (N,)

        ds = Rellis3DDataset("/data/RELLIS", keys=["lidar", "poses"])
        # sample.data["poses"]  -> np.ndarray (3, 4)
    """

    _profile = "rellis.yaml"
