import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d

# Voxel construction using Poison surface reconstruction

def plot(image, index, title):
    plt.subplot(1, 2, index)
    plt.imshow(image)
    plt.title(title)

def sharpen_image(image):
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened

def get_depth_map(image):
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    input_batch = transform(image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

def remove_background(image, depth_map, threshold=0.5):
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    mask = depth_map > threshold
    mask = np.stack([mask] * 3, axis=-1)
    result = np.zeros_like(image)
    result[mask] = image[mask]
    return result

def create_point_cloud(image, depth_map):
    h, w = depth_map.shape
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    points = []
    colors = []
    black_threshold = 10
    for v in range(h):
        for u in range(w):
            r, g, b = image[v, u]
            if r < black_threshold and g < black_threshold and b < black_threshold:
                continue
            z = depth_normalized[v, u]
            x = (u - w / 2) / w
            y = (v - h / 2) / h
            points.append([x, y, z])
            colors.append([r / 255, g / 255, b / 255])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def voxelize_point_cloud(pcd, voxel_size=0.01):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def voxel_grid_to_mesh(voxel_grid):
    # Extract voxel centers
    voxel_centers = []
    voxel_colors = []
    for voxel in voxel_grid.get_voxels():
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        voxel_centers.append(center)
        voxel_colors.append(voxel.color)
    
    # Create point cloud from voxel centers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
    pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))
    
    # Estimate normals - required before BPA
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)
    
    # Compute average distance between points to estimate radius parameter
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    
    # Create multiple radii for multi-scale ball pivoting
    radii = [avg_distance * 2, avg_distance * 4, avg_distance * 8]
    
    # Apply Ball-Pivoting Algorithm
    print("Creating mesh using Ball-Pivoting Algorithm...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    # Clean the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Fill holes (optional)
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    
    return mesh

if __name__ == "__main__":
    image_path = 'Assets/cat.jpg'
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    print(f"Debug: Image loaded successfully, shape: {image.shape}")

    max_dimension = 640
    if max(image.shape) > max_dimension:
        scale = max_dimension / max(image.shape[0], image.shape[1])
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)
        print(f"Debug: Resized image to {image.shape}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Debug: Converting color space and sharpening")
    image = sharpen_image(image)

    print("Debug: Generating depth map")
    depth_map = get_depth_map(image)
    print(f"Debug: Depth map generated, shape: {depth_map.shape}")

    # Depth Map Interpolation (Increase resolution)
    scale_factor = 2
    new_size = (depth_map.shape[1] * scale_factor, depth_map.shape[0] * scale_factor)
    depth_map_interpolated = cv2.resize(depth_map, new_size, interpolation=cv2.INTER_CUBIC)

    output_image = remove_background(image, depth_map, threshold=0.5)
    depth_output_image = get_depth_map(output_image)

    plt.figure(figsize=(10, 5))
    plot(output_image, 1, "No Background")
    plt.show()

    pcd = create_point_cloud(output_image, depth_output_image)

    o3d.visualization.draw_geometries([pcd])

    voxel_size = 0.01
    voxel_grid = voxelize_point_cloud(pcd, voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])

    mesh = voxel_grid_to_mesh(voxel_grid)

    # Mesh Smoothing
    smoothed_mesh = mesh.filter_smooth_simple(number_of_iterations=20)
    o3d.visualization.draw_geometries([smoothed_mesh])

    # Mesh Subdivision
    subdivided_mesh = mesh.subdivide_loop(number_of_iterations=1)
    o3d.visualization.draw_geometries([subdivided_mesh])