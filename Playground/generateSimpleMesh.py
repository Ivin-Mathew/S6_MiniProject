import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import sys
import traceback

def sharpen_image(image):
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    smoothed = cv2.GaussianBlur(sharpened, (3, 3), 0)
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

def create_point_cloud(image, depth_map):
    h, w = depth_map.shape
    # Normalize depth map for better visualization and point cloud generation
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Create point cloud
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            z = depth_normalized[v, u]  # Use normalized depth as z-coordinate
            x = (u - w / 2) / w  # Normalize x and y to be between -0.5 and 0.5 for better visualization
            y = (v - h / 2) / h
            points.append([x, y, z])
            # Use image color for point color
            r, g, b = image[v, u]
            colors.append([r / 255, g / 255, b / 255])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pcd


if __name__ == "__main__":
    # ... (Image loading and depth map generation as before) ...
    
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load and process image
    image_path = 'Assets/cat.jpg'
    print(f"Debug: Loading image from {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    print(f"Debug: Image loaded successfully, shape: {image.shape}")
    
    # Optionally resize image if it's too large
    max_dimension = 640
    if max(image.shape) > max_dimension:
        scale = max_dimension / max(image.shape[0], image.shape[1])
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)
        print(f"Debug: Resized image to {image.shape}")
    
    # Convert BGR to RGB and process
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Debug: Converting color space and sharpening")
    image = sharpen_image(image)
    
    # Generate depth map
    print("Debug: Generating depth map")
    depth_image = get_depth_map(image)
    print(f"Debug: Depth map generated, shape: {depth_image.shape}")
    
    # Visualize depth map
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(depth_image, cmap='plasma')
    plt.title("Depth Map")
    plt.colorbar()
    plt.show()
    # Create point cloud
    pcd = create_point_cloud(image, depth_image)

    # Visualize point cloud in Open3D
    o3d.visualization.draw_geometries([pcd])