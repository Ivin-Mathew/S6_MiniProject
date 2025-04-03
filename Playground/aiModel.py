import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d

# Helper functions
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

def upsample_point_cloud_with_ai(pcd, target_points=100000):
    """
    Upsample the point cloud using a pre-trained PU-Net model.
    """
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(pcd.points)
    points = points[np.newaxis, :, :]  # Add batch dimension
    points = torch.tensor(points, dtype=torch.float32)

    # Load a pre-trained PU-Net model (replace with actual model loading)
    class PUNet(torch.nn.Module):
        def __init__(self):
            super(PUNet, self).__init__()
            # Define the model architecture (simplified for demonstration)
            self.fc = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    model = PUNet()
    model.load_state_dict(torch.load("path_to_pretrained_weights.pth", map_location="cpu"))
    model.eval()

    # Upsample the point cloud
    with torch.no_grad():
        upsampled_points = model(points)

    # Convert back to Open3D point cloud
    upsampled_pcd = o3d.geometry.PointCloud()
    upsampled_pcd.points = o3d.utility.Vector3dVector(upsampled_points.squeeze().numpy())
    upsampled_pcd.colors = pcd.colors  # Retain original colors
    return upsampled_pcd

def create_mesh_from_point_cloud(pcd):
    """
    Create a mesh from the point cloud using Poisson Surface Reconstruction.
    """
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Optional: Smooth the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    return mesh

if __name__ == "__main__":
    # Load and preprocess the image
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

    # Generate depth map
    print("Debug: Generating depth map")
    depth_map = get_depth_map(image)
    print(f"Debug: Depth map generated, shape: {depth_map.shape}")

    # Remove background
    output_image = remove_background(image, depth_map, threshold=0.5)
    depth_output_image = get_depth_map(output_image)

    # Create initial point cloud
    pcd = create_point_cloud(output_image, depth_output_image)

    # Upsample the point cloud using AI
    print("Debug: Upsampling point cloud with AI")
    upsampled_pcd = upsample_point_cloud_with_ai(pcd)

    # Create mesh from the upsampled point cloud
    print("Debug: Creating mesh from upsampled point cloud")
    mesh = create_mesh_from_point_cloud(upsampled_pcd)

    # Visualize the final mesh
    o3d.visualization.draw_geometries([mesh])