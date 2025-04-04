
import cv2
import numpy as np
import open3d as o3d
from .common_utils import *

def voxelize_point_cloud(pcd, voxel_size=0.01):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def voxel_grid_to_mesh(voxel_grid):
    voxel_centers = []
    voxel_colors = []
    for voxel in voxel_grid.get_voxels():
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        voxel_centers.append(center)
        voxel_colors.append(voxel.color)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
    pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)
    
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False
    )
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    return mesh