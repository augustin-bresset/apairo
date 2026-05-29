from apairo.core.profiled_dataset import ProfiledDataset


class Goose3DDataset(ProfiledDataset):
    """GOOSE 3D dataset -- outdoor off-road LiDAR with traversability labels.

    Keys: ``lidar`` (float32, shape (N, 4)), ``labels`` (int64).
    Split: ``"train"``, ``"val"``, or ``"test"``.

    Example::

        ds = Goose3DDataset("/data/GOOSE_3D", keys=["lidar", "labels"], split="train")
        sample = ds[0]
        # sample.data["lidar"]  -> np.ndarray (N, 4)
        # sample.data["labels"] -> np.ndarray (N,)
    """

    _profile = "goose.yaml"
