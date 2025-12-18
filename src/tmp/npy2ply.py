import open3d as o3d
import numpy as np


xyz = np.load("data/tartan_2_kitti/example1/velodyne_0/000020.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

o3d.io.write_point_cloud("tmp/lidar_velo20.ply", pcd)