import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import open3d as o3d

def plot(image, index, title):
    plt.subplot(1, 2, index)
    plt.imshow(image)
    plt.title(title)

def sharpen_image(image):
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened

def get_depth_map(image):
    model_type = "MiDaS_small"
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

    # Normalize depth map
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    # Create a binary mask for foreground (closer objects)
    mask = depth_map > threshold  # Keep closer objects

    # Convert the mask to a 3-channel mask
    mask = np.stack([mask] * 3, axis=-1)

    # Apply the mask to retain only the foreground
    result = np.zeros_like(image)
    result[mask] = image[mask]
    return result



def point_cloud_to_mesh(pcd, method="poisson"):
    """
    Convert a point cloud to a mesh using Poisson or Ball-Pivoting Algorithm.
    :param pcd: Open3D point cloud object
    :param method: "poisson" or "bpa" (Ball-Pivoting Algorithm)
    :return: Open3D triangle mesh object
    """
    pass


def create_point_cloud(image, depth_map):
    h, w = depth_map.shape
    # Normalize depth map for better visualization and point cloud generation
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # Create point cloud
    points = []
    colors = []
    black_threshold = 0  # Define the threshold for black pixels
    
    for v in range(h):
        for u in range(w):
            r, g, b = image[v, u]
            # Ignore black pixels (background)
            if r < black_threshold and g < black_threshold and b < black_threshold:
                continue  # Skip black pixels

            z = depth_normalized[v, u]  # Use normalized depth as z-coordinate
            x = (u - w / 2) / w  # Normalize x and y to be between -0.5 and 0.5 for better visualization
            y = (v - h / 2) / h
            points.append([x, y, -z])
            colors.append([r / 255, g / 255, b / 255])  # Normalize RGB values

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pcd



if __name__ == "__main__":
    image_path = '../Assets/1.webp'
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    print(f"Debug: Image loaded successfully, shape: {image.shape}")
    
    # Resize if too large
    max_dimension = 640
    if max(image.shape) > max_dimension:
        scale = max_dimension / max(image.shape[0], image.shape[1])
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)
        print(f"Debug: Resized image to {image.shape}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Debug: Converting color space and sharpening")
    image = sharpen_image(image)
    
    # Generate depth map
    print("Debug: Generating depth map")
    depth_map = get_depth_map(image)
    print(f"Debug: Depth map generated, shape: {depth_map.shape}")
 
    # Remove background using depth map
    output_image = remove_background(image, depth_map, threshold=0.5)
    depth_output_image = get_depth_map(output_image)
   
    # Visualize results
    plt.figure(figsize=(10, 5))
    plot(output_image, 1, "No Background")
    plt.show()
    
    # Create point cloud
    pcd = create_point_cloud(output_image, depth_output_image)

    # Visualize point cloud in Open3D
    o3d.visualization.draw_geometries([pcd])
    
    
    mesh = point_cloud_to_mesh(pcd)
    o3d.visualization.draw_geometries([mesh])
