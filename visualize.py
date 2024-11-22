import open3d as o3d

def visualize_ply(file_path, output_image_path):
    # Load the .ply file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Print some information about the point cloud
    print(pcd)
    print("Number of points:", len(pcd.points))
    
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    
    # Render the point cloud and capture the image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_image_path)
    vis.destroy_window()

if __name__ == "__main__":
    # Replace 'path/to/your/file.ply' with the actual path to your .ply file
    file_path = '/home/lbocchi/UnA-Gen/outputs/train/courtyard_laceShoe_00/ao_cano/mesh_epoch_109.ply'
    output_image_path = '/home/lbocchi/UnA-Gen/outputs/train/courtyard_laceShoe_00/ao_cano/exports/mesh_epoch_109.png'
    visualize_ply(file_path, output_image_path)