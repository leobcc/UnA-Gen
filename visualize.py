import trimesh
from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import Meshes, VariableTopologyMeshes
import glob
import argparse
import imageio
from aitviewer.headless import HeadlessRenderer
import os

def vis_dynamic(args):
    vertices = []
    faces = []
    vertex_normals = []
    deformed_mesh_paths = sorted(glob.glob(f'{args.path}/*_deformed.ply'))
    for deformed_mesh_path in deformed_mesh_paths:
        mesh = trimesh.load(deformed_mesh_path, process=False)
        # center the human
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        vertices.append(mesh.vertices)
        faces.append(mesh.faces)
        vertex_normals.append(mesh.vertex_normals)

    meshes = VariableTopologyMeshes(vertices,
                                    faces,
                                    vertex_normals,
                                    preload=True 
                                    )

    meshes.norm_coloring = True
    meshes.flat_shading = True
    viewer = HeadlessRenderer()
    viewer.scene.add(meshes)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = True
    viewer.save_video(video_dir=os.path.join("outputs", "video.mp4"))

def vis_static(args):
    mesh = trimesh.load(args.path, process=False)
    mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='mesh', flat_shading=True)
    mesh.norm_coloring = True
    viewer = HeadlessRenderer()
    viewer.scene.add(mesh)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = True
    viewer.save_video(video_dir=os.path.join("outputs", "video.mp4"))

def vis_dynamic_canonical_train(args):
    vertices = []
    faces = []
    vertex_normals = []
    deformed_mesh_paths = sorted(glob.glob(f'{args.path}/*.ply'))
    for deformed_mesh_path in deformed_mesh_paths:
        mesh = trimesh.load(deformed_mesh_path, process=False)
        # center the human
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        vertices.append(mesh.vertices)
        faces.append(mesh.faces)
        vertex_normals.append(mesh.vertex_normals)

    meshes = VariableTopologyMeshes(vertices,
                                    faces,
                                    vertex_normals,
                                    preload=True 
                                    )

    meshes.norm_coloring = True
    meshes.flat_shading = True
    viewer = HeadlessRenderer()
    viewer.scene.add(meshes)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = True
    viewer.save_video(video_dir=os.path.join("outputs", "video.mp4"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Visualization')
    parser.add_argument('--mode', type=str, help='mode: static, dynamic, or dynamic_canonical_train')
    parser.add_argument('--path', type=str, help='path to the file')
    args = parser.parse_args()
    
    if args.mode == 'static':
        vis_static(args)
    elif args.mode == 'dynamic':
        vis_dynamic(args)
    elif args.mode == 'dynamic_canonical_train':
        vis_dynamic_canonical_train(args)
