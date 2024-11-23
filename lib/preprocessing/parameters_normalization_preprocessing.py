import numpy as np
import yaml
import torch
import os
from preprocessing_utils import (smpl_to_pose, PerspectiveCamera, Renderer, render_trimesh, \
                                estimate_translation_cv2, transform_smpl)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/home/lbocchi/UnA-Gen/data/data/train/courtyard_laceShoe_00/cameras.npz'

cameras_old_format = np.load(path, allow_pickle=True)
print("cameras_old_format:", cameras_old_format['cam_0'])

path = '/home/lbocchi/UnA-Gen/data/data/train/0012_09/camera_1/camera_params.pkl'

cameras_smplx = np.load(path, allow_pickle=True)
print("camera keys:", cameras_smplx.keys())
print("D:", cameras_smplx['D'])

K = cameras_smplx['K']
RT = cameras_smplx['RT']
    
K4x4 = np.eye(4)
K4x4[:3, :3] = K

#P = K4x4 @ RT

metadata_path = '/home/lbocchi/UnA-Gen/data/data/train/0012_09/metadata.yaml'
with open(metadata_path, 'r') as file:
    metadata = yaml.safe_load(file)

number_of_frames = metadata['end_frame'] - metadata['start_frame']

video_folder = '/home/lbocchi/UnA-Gen/data/data/train/0012_09'
smplx_data = np.load(os.path.join(video_folder, "smplx.npy"), allow_pickle=True).item()
smpl_shape = smplx_data['betas'][0]
fullpose = smplx_data['fullpose']
smpl_trans_all = smplx_data['transl']
from smpl import SMPL
smpl_model = SMPL('/home/lbocchi/UnA-Gen/smpl/smpl_model', gender=metadata['gender']).cuda()

cam_extrinsics = np.eye(4)
cam_extrinsics = RT.copy()

output_trans = []
output_pose = []
camera_projection_matrices = {}
for idx in range(number_of_frames):
    T_hip = smpl_model.get_T_hip(betas=torch.tensor(smpl_shape)[None].float().cuda()).squeeze().cpu().numpy()

    target_extrinsic = np.eye(4)
    target_extrinsic[1:3] *= -1
    smpl_pose = fullpose[idx].copy()
    smpl_pose = smpl_pose.reshape(-1)[:72]
    target_extrinsic, smpl_pose, smpl_trans = transform_smpl(cam_extrinsics, target_extrinsic, smpl_pose, smpl_trans_all[idx], T_hip)
    smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float().to(device),
                                body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                global_orient=torch.tensor(smpl_pose[:3])[None].float().to(device),
                                transl=torch.tensor(smpl_trans)[None].float().to(device))
    smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

    # we need to center the human for every frame due to the potentially large global movement
    v_max = smpl_verts.max(axis=0)
    v_min = smpl_verts.min(axis=0)
    normalize_shift = -(v_max + v_min) / 2.

    trans = smpl_trans + normalize_shift
    output_trans.append(trans)
    output_pose.append(smpl_pose)
    
    target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)

    P = K4x4 @ target_extrinsic

    camera_projection_matrices['cam_'+str(idx)] = P
    #camera_projection_matrices['K_'+str(idx)] = K
    #camera_projection_matrices['RT_'+str(idx)] = RT

np.savez('/home/lbocchi/UnA-Gen/data/data/train/0012_09/camera_1/camera_params.npz', **camera_projection_matrices)
np.save(os.path.join('/home/lbocchi/UnA-Gen/data/data/train/0012_09/camera_1/', 'normalize_trans.npy'), np.array(output_trans))
np.save(os.path.join('/home/lbocchi/UnA-Gen/data/data/train/0012_09/camera_1/', 'poses.npy'), np.array(output_pose))

path_new_cameras = '/home/lbocchi/UnA-Gen/data/data/train/0012_09/camera_1/camera_params.npz'
cameras_new = np.load(path_new_cameras, allow_pickle=True)
#print("cameras_new:", cameras_new['cam_0'])