# 3DPW Dataset Preprocessing:
# Preprocessing the generate files containing SMPL transformations, pose estimation, shape estimation, and camera intrinsics
# After running this script, the initial data will be divided in frame-based files

import os
import glob
import cv2
import numpy as np
import torch
import yaml
from lib.utils import utils  
from smpl_server import SMPLServer
from deformer import SMPLDeformer

def preprocess_data(video_folder):
    with open(os.path.join(video_folder, "metadata.yaml"), 'r') as file:
        metadata = yaml.safe_load(file)
    gender = metadata['gender']
    training_indices = list(range(metadata['start_frame'], metadata['end_frame']))
    betas = np.load(os.path.join(video_folder, "mean_shape.npy"))
    body_model_params = {}
    body_model_params['betas'] = torch.tensor(np.load(os.path.join(video_folder, "mean_shape.npy"))[None], dtype=torch.float32)
    body_model_params['transl'] = torch.tensor(np.load(os.path.join(video_folder, 'normalize_trans.npy'))[training_indices], dtype=torch.float32)
    body_model_params['global_orient'] = torch.tensor(np.load(os.path.join(video_folder, 'poses.npy'))[training_indices][:, :3], dtype=torch.float32)
    body_model_params['body_pose'] = torch.tensor(np.load(os.path.join(video_folder, 'poses.npy'))[training_indices] [:, 3:], dtype=torch.float32)

    smpl_params_folder = os.path.join(video_folder, "smpl_params")
    smpl_server = SMPLServer(gender=gender, betas=betas)
    deformer = SMPLDeformer(betas=betas, gender=gender)
    
    smpl_tfs_folder = os.path.join(video_folder, "smpl_tfs")
    os.makedirs(smpl_tfs_folder, exist_ok=True)

    for filename in os.listdir(smpl_params_folder):
        filepath = os.path.join(smpl_params_folder, filename)
        smpl_params = torch.load(filepath)
        smpl_params = smpl_params.unsqueeze(0)

        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(smpl_params, [1, 3, 72, 10], dim=1)
        scale = scale.cuda()
        smpl_trans = smpl_trans.cuda()
        smpl_pose = smpl_pose.cuda()
        smpl_shape = smpl_shape.cuda()

        #smpl_shape = body_model_params['betas'] if body_model_params['betas'].dim() == 2 else body_model_params['betas'].unsqueeze(0)
        #smpl_trans = body_model_params['transl']
        #smpl_pose = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        
        smpl_outputs = smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs['smpl_tfs']
        smpl_tfs = smpl_tfs.cpu()
        
        torch.save(smpl_tfs, os.path.join(smpl_tfs_folder, filename))

    return

if __name__ == "__main__":
    video_folder = '/UnA-Gen/data/data/train/courtyard_laceShoe_00'
    preprocess_data(video_folder)
    