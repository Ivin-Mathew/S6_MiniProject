import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import os

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
            points.append([x, y, -z])
            colors.append([r / 255, g / 255, b / 255])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def export_mesh_to_obj(mesh, file_name, texture=None):
    """
    Exports a 3D mesh to an OBJ file along with its MTL and texture files.

    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to export.
        file_name (str): The full path (without extension) to save the OBJ file.
        texture (np.ndarray, optional): The texture image to save alongside the OBJ file.
    """
    obj_file = f"{file_name}.obj"
    mtl_file = f"{file_name}.mtl"
    texture_file = f"{file_name}.png"

    mesh.compute_vertex_normals()
    # Save the mesh to an OBJ file
    o3d.io.write_triangle_mesh(obj_file, mesh, write_triangle_uvs=True)

    # If a texture is provided, save it and create the MTL file
    if texture is not None:
        import cv2
        cv2.imwrite(texture_file, texture)
        print(f"Texture saved to {texture_file}")

        # Create the MTL file
        with open(mtl_file, "w") as mtl:
            mtl.write(f"newmtl material_0\n")
            mtl.write(f"Ka 1.000 1.000 1.000\n")
            mtl.write(f"Kd 1.000 1.000 1.000\n")
            mtl.write(f"Ks 0.000 0.000 0.000\n")
            mtl.write(f"d 1.0\n")
            mtl.write(f"illum 2\n")
            mtl.write(f"map_Kd {os.path.basename(texture_file)}\n")
        print(f"MTL file saved to {mtl_file}")

        # Update the OBJ file to reference the MTL file
        with open(obj_file, "r") as obj:
            obj_data = obj.readlines()
        with open(obj_file, "w") as obj:
            obj.write(f"mtllib {os.path.basename(mtl_file)}\n")
            obj.writelines(obj_data)

    print(f"Mesh exported to {obj_file}")
    
def voxel_grid_to_cube_mesh(voxel_grid):
    """
    Converts a VoxelGrid to a mesh where each voxel is represented as a cube with UV mapping.

    Args:
        voxel_grid (o3d.geometry.VoxelGrid): The voxel grid to convert.

    Returns:
        o3d.geometry.TriangleMesh: A mesh representing the voxel grid as cubes with UV mapping.
    """
    cube_mesh = o3d.geometry.TriangleMesh()

    # Iterate through all voxels in the grid
    for voxel in voxel_grid.get_voxels():
        # Get the center of the voxel
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

        # Create a cube mesh for the voxel
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_grid.voxel_size,
                                                    height=voxel_grid.voxel_size,
                                                    depth=voxel_grid.voxel_size)

        # Translate the cube to the voxel's center
        cube.translate(center)

        # Assign UV coordinates to the cube
        cube.compute_vertex_normals()
        cube.triangle_uvs = o3d.utility.Vector2dVector([
            [0, 0], [1, 0], [1, 1], [0, 1],  # Front face
            [0, 0], [1, 0], [1, 1], [0, 1],  # Back face
            [0, 0], [1, 0], [1, 1], [0, 1],  # Left face
            [0, 0], [1, 0], [1, 1], [0, 1],  # Right face
            [0, 0], [1, 0], [1, 1], [0, 1],  # Top face
            [0, 0], [1, 0], [1, 1], [0, 1],  # Bottom face
        ])

        # Optionally, assign the voxel's color to the cube
        cube.paint_uniform_color(voxel.color)

        # Combine the cube with the overall mesh
        cube_mesh += cube

    # Simplify the mesh by merging vertices
    cube_mesh.merge_close_vertices(1e-6)

    return cube_mesh