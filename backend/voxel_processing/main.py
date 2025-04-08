from .voxelBPA import *
from .voxelPoisonSurface import *
import os

def process_image(image_path, output_folder):
    """
    Processes an image to generate .obj, .mtl, and texture files.

    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder to save the output files.

    Returns:
        tuple: Paths to the generated .obj, .mtl, and texture files.
    """
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

    output_image = remove_background(image, depth_map, threshold=0.5)
    depth_output_image = get_depth_map(output_image)

    pcd = create_point_cloud(output_image, depth_output_image)

    # Voxel generation
    voxel_size = 0.01
    voxel_grid = voxelize_point_cloud(pcd, voxel_size)
    
    cube_mesh = voxel_grid_to_cube_mesh(voxel_grid)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Export the mesh to .obj, .mtl, and texture files
    obj_path = os.path.join(output_folder, "mesh.obj")
    mtl_path = os.path.join(output_folder, "mesh.mtl")
    texture_path = os.path.join(output_folder, "mesh.png")
    export_mesh_to_obj(cube_mesh, obj_path[:-4], texture=image)

    return obj_path, mtl_path, texture_path