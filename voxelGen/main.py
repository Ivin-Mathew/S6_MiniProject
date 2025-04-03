# filepath: S6_MiniProject/src/main.py
from modules.voxelBPA import *
from modules.voxelPoisonSurface import *

if __name__ == "__main__":
    image_path = 'Assets/bird.png'
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

    plt.figure(figsize=(10, 5))
    plot(output_image, 1, "No Background")
    # plt.show()

    pcd = create_point_cloud(output_image, depth_output_image)

    # o3d.visualization.draw_geometries([pcd])

    #voxel generation
    voxel_size = 0.01
    voxel_grid = voxelize_point_cloud(pcd, voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])
    
    cube_mesh = voxel_grid_to_cube_mesh(voxel_grid)
    export_mesh_to_obj(cube_mesh, "mesh", texture=image)