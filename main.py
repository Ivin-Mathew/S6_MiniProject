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



def create_point_cloud(depth_map, image):
    """
    Generate a 3D point cloud from a depth map and image with detailed debugging.
    """
    try:
        print("Debug: Starting point cloud creation")
        print(f"Debug: Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")
        print(f"Debug: Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Validate input dimensions
        if depth_map.shape[:2] != image.shape[:2]:
            raise ValueError(f"Dimension mismatch: depth_map {depth_map.shape[:2]} != image {image.shape[:2]}")

        # Ensure image is in correct format (uint8)
        print("Debug: Converting image to uint8")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Normalize and scale depth map
        print("Debug: Normalizing depth map")
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        print(f"Debug: Depth range - min: {depth_min}, max: {depth_max}")
        
        # Check for invalid depth values
        if np.isnan(depth_map).any():
            print("Warning: NaN values found in depth map")
        if np.isinf(depth_map).any():
            print("Warning: Infinite values found in depth map")
            
        depth_map_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)
        print(f"Debug: Normalized depth map - min: {np.min(depth_map_normalized)}, max: {np.max(depth_map_normalized)}")

        # Create Open3D images
        print("Debug: Creating Open3D images")
        try:
            color_o3d = o3d.geometry.Image(image)
            depth_o3d = o3d.geometry.Image(depth_map_normalized)
            print("Debug: Successfully created Open3D images")
        except Exception as e:
            print(f"Error creating Open3D images: {str(e)}")
            raise

        # Create RGBD image
        print("Debug: Creating RGBD image")
        try:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=65535.0,
                depth_trunc=65535.0,
                convert_rgb_to_intensity=False
            )
            print("Debug: Successfully created RGBD image")
        except Exception as e:
            print(f"Error creating RGBD image: {str(e)}")
            raise

        # Calculate intrinsic parameters
        height, width = depth_map.shape
        fx = fy = max(width, height)
        cx, cy = width / 2, height / 2
        
        print(f"Debug: Creating intrinsic parameters - width: {width}, height: {height}, fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        # Create point cloud
        print("Debug: Creating point cloud from RGBD image")
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        print(f"Debug: Initial point cloud created with {len(pcd.points)} points")

        # Remove invalid points
        print("Debug: Cleaning point cloud")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        valid_points = ~np.any(np.isnan(points), axis=1)
        valid_points &= ~np.any(np.isinf(points), axis=1)
        valid_points &= np.linalg.norm(points, axis=1) > 0

        pcd.points = o3d.utility.Vector3dVector(points[valid_points])
        pcd.colors = o3d.utility.Vector3dVector(colors[valid_points])
        print(f"Debug: Clean point cloud has {len(pcd.points)} points")

        # Transform point cloud
        print("Debug: Transforming point cloud")
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return pcd

    except Exception as e:
        print("Error in create_point_cloud:")
        print(traceback.format_exc())
        raise

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import sys
import traceback

def create_point_cloud_alternative(depth_map, image):
    """
    Alternative method to generate point cloud using direct computation.
    """
    try:
        print("Debug: Starting alternative point cloud creation")
        
        height, width = depth_map.shape
        
        # Create coordinate grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Define camera intrinsics
        fx = fy = max(width, height)
        cx, cy = width / 2, height / 2
        
        # Normalize depth map to a reasonable range (e.g., 0-10 meters)
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min) * 10.0
        
        # Calculate 3D coordinates
        z = depth_normalized
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        
        # Reshape to points array
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)
        
        # Get colors
        colors = image.reshape(-1, 3) / 255.0
        
        # Create Open3D point cloud
        print("Debug: Creating Open3D point cloud object")
        pcd = o3d.geometry.PointCloud()
        
        # Remove invalid points
        valid_points = ~np.any(np.isnan(points), axis=1)
        valid_points &= ~np.any(np.isinf(points), axis=1)
        valid_points &= np.linalg.norm(points, axis=1) > 0
        
        points = points[valid_points]
        colors = colors[valid_points]
        
        print(f"Debug: Setting {len(points)} valid points")
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Optional: Estimate normals for better visualization
        print("Debug: Estimating normals")
        try:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))
            print("Debug: Normals estimated successfully")
        except Exception as e:
            print(f"Warning: Could not estimate normals: {str(e)}")
        
        # Optional: Downsample to reduce density
        print("Debug: Downsampling point cloud")
        try:
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            print(f"Debug: Downsampled to {len(pcd.points)} points")
        except Exception as e:
            print(f"Warning: Could not downsample: {str(e)}")
        
        print(f"Debug: Final point cloud has {len(pcd.points)} points")
        return pcd
        
    except Exception as e:
        print("Error in create_point_cloud_alternative:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        # Print versions for debugging
        print(f"Open3D version: {o3d.__version__}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Load and process image
        image_path = 'Assets/bird.jpg'
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
        
        # Create point cloud using alternative method
        print("Debug: Creating point cloud using alternative method")
        pcd = create_point_cloud_alternative(depth_image, image)
        
        # Save the point cloud before visualization (as backup)
        print("Debug: Saving point cloud to file")
        o3d.io.write_point_cloud("output_cloud.ply", pcd)
        
        # Simple visualization
        print("Debug: Visualizing point cloud")
        try:
            # Create coordinate frame for reference
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[0, 0, 0])
            
            # Visualize using basic draw_geometries
            o3d.visualization.draw_geometries(
                [pcd, coordinate_frame]
            )
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            print("The point cloud has been saved to 'output_cloud.ply'")
            print("You can open it with another 3D viewer if the visualization fails")
        
    except Exception as e:
        print("Error in main:")
        print(traceback.format_exc())
        sys.exit(1)