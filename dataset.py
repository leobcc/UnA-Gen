import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from smpl_server import SMPLServer
from torchvision.utils import save_image

class una_gen_dataset(Dataset):
    def __init__(self, data_dir, split='train', frame_skip=1, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data = []

        split_dir = os.path.join(data_dir, split)
        for video_folder in os.listdir(split_dir):
            if video_folder == 'courtyard_backpack_00':   # This is a temporary fix, we should remove this line
                continue
            video_path = os.path.join(split_dir, video_folder)
            frame_dir = os.path.join(video_path, "rgb_frames")
            mask_dir = os.path.join(video_path, "masks")
            smpl_params_dir = os.path.join(video_path, "smpl_params")
            pose_dir = os.path.join(video_path, "pose")
            intrinsics_dir = os.path.join(video_path, "intrinsics")
            smpl_tfs_dir = os.path.join(video_path, "smpl_tfs")

            betas_path = os.path.join(video_path, "mean_shape.npy")
            metadata_path = os.path.join(video_path, "metadata.yaml")

            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)

            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
            smpl_params_files = sorted([f for f in os.listdir(smpl_params_dir) if f.endswith(".pt")])
            pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".pt")])
            intrinsics_files = sorted([f for f in os.listdir(intrinsics_dir) if f.endswith(".pt")])
            smpl_tfs_files = sorted([f for f in os.listdir(smpl_tfs_dir) if f.endswith(".pt")])

            num_frames = len(frame_files)
            num_masks = len(mask_files)
            num_smpl_params = len(smpl_params_files)
            num_poses = len(pose_files)
            num_intrinsics = len(intrinsics_files)
            num_smpl_tfs = len(smpl_tfs_files)
            if num_frames != num_masks or num_frames != num_smpl_params or num_frames != num_poses or num_frames != num_intrinsics or num_frames != num_smpl_tfs:
                raise ValueError(f"Number of frames ({num_frames}) does not match number of masks ({num_masks}) or the other data ({num_smpl_params}, {num_poses}, {num_intrinsics}, {num_smpl_tfs}) for video {video_folder}")

            for i, (frame_file, mask_file, smpl_params_file, pose_file, intrinsics_file, smpl_tfs_file) in enumerate(zip(frame_files, mask_files, smpl_params_files, pose_files, intrinsics_files, smpl_tfs_files)):
                if i % frame_skip == 0:
                    frame_path = os.path.join(frame_dir, frame_file)
                    mask_path = os.path.join(mask_dir, mask_file)
                    smpl_params_path = os.path.join(smpl_params_dir, smpl_params_file)
                    pose_path = os.path.join(pose_dir, pose_file)
                    intrinsics_path = os.path.join(intrinsics_dir, intrinsics_file)
                    smpl_tfs_path = os.path.join(smpl_tfs_dir, smpl_tfs_file)
                    self.data.append({'frame_path': frame_path, 'mask_path': mask_path, 'smpl_params_path': smpl_params_path, 'pose_path': pose_path, 'intrinsics_path': intrinsics_path, 'smpl_tfs_path': smpl_tfs_path, 'betas_path': betas_path, 'metadata': metadata})

                ''' The smpl tfs are already preprocessed
                betas = torch.tensor(np.load(betas_path)[None], dtype=torch.float32)
                smpl_server = SMPLServer(gender=metadata['gender'], betas=betas)
                smpl_params = torch.load(smpl_params_path)
                print("smpl_params shape:", smpl_params.shape)
                scale, smpl_trans, smpl_pose, smpl_shape = torch.split(smpl_params, [1, 3, 72, 10], dim=0)
                smpl_output = smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)   #  invokes the SMPL model to obtain the transformations for pose and shape changes

                smpl_tfs = smpl_output['smpl_tfs']
                torch.save(smpl_tfs, os.path.join(video_path, "smpl_tfs", f"{frame_file}.pt"))
                '''

            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_info = self.data[idx]
        frame = cv2.imread(frame_info['frame_path'])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(frame_info['mask_path'], cv2.IMREAD_GRAYSCALE)
        betas = torch.tensor(np.load(frame_info['betas_path'])[None], dtype=torch.float32) # Needed?
        metadata = frame_info['metadata']
        
        masked_image = cv2.bitwise_and(frame, frame, mask=mask)
        # I think we should not resize
        # masked_image = cv2.resize(masked_image, (1080, 1080))   # When reshaping we have to also change the parameters etc. run in the preprocessing of the original image
        masked_image = transforms.ToTensor()(masked_image)

        smpl_params = torch.load(frame_info['smpl_params_path']) # Needed?
        pose = torch.load(frame_info['pose_path']) # Needed?
        intrinsics = torch.load(frame_info['intrinsics_path']) # Needed?
        smpl_tfs = torch.load(frame_info['smpl_tfs_path'])

        inputs = {'masked_image': masked_image, 
                  'smpl_params': smpl_params, 
                  'pose': pose, 
                  'intrinsics': intrinsics,
                  'smpl_tfs': smpl_tfs,
                  'betas': betas,
                  'metadata': metadata}

        return inputs
