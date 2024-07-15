import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from smpl_server import SMPLServer
from torchvision.utils import save_image
from lib.utils.utils import get_camera_params

from torchvision.transforms import Compose
import sys
sys.path.append('/UnA-Gen/supp_repos/Depth_Anything_main/')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class una_gen_dataset(Dataset):
    def __init__(self, data_dir, split='train', frame_skip=1, image_size=(512,512), crop=True, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.crop = crop
        self.transform = transform
        self.data = []

        split_dir = os.path.join(data_dir, split)
        for video_folder in os.listdir(split_dir):
            if video_folder == 'courtyard_backpack_00':   # This is a temporary fix, we should remove this line
                continue
            video_path = os.path.join(split_dir, video_folder)
            frame_dir = os.path.join(video_path, "rgb_frames")
            mask_dir = os.path.join(video_path, "masks")
            masked_img_dir = os.path.join(video_path, "masked_img")
            depth_img_dir = os.path.join(video_path, "depth_img_bw")   # Change to depth_img if we want to use the 3-channels depth images
            smpl_params_dir = os.path.join(video_path, "smpl_params")
            pose_dir = os.path.join(video_path, "pose")
            intrinsics_dir = os.path.join(video_path, "intrinsics")
            smpl_tfs_dir = os.path.join(video_path, "smpl_tfs")
            smpl_outputs_dir = os.path.join(video_path, "smpl_outputs")
            smpl_verts_cano_dir = os.path.join(video_path, "smpl_verts_cano")

            betas_path = os.path.join(video_path, "mean_shape.npy")
            metadata_path = os.path.join(video_path, "metadata.yaml")

            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)

            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
            masked_images = sorted([f for f in os.listdir(masked_img_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
            depth_images = sorted([f for f in os.listdir(depth_img_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
            smpl_params_files = sorted([f for f in os.listdir(smpl_params_dir) if f.endswith(".pt")])
            pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".pt")])
            intrinsics_files = sorted([f for f in os.listdir(intrinsics_dir) if f.endswith(".pt")])
            smpl_tfs_files = sorted([f for f in os.listdir(smpl_tfs_dir) if f.endswith(".pt")])
            smpl_outputs_files = sorted([f for f in os.listdir(smpl_outputs_dir) if f.endswith(".pt")])
            smpl_verts_cano_files = sorted([f for f in os.listdir(smpl_verts_cano_dir) if f.endswith(".pt")])

            num_frames = len(frame_files)
            num_masks = len(mask_files)
            num_masked_images = len(masked_images)
            num_depth_images = len(depth_images)
            num_smpl_params = len(smpl_params_files)
            num_poses = len(pose_files)
            num_intrinsics = len(intrinsics_files)
            num_smpl_tfs = len(smpl_tfs_files)
            num_smpl_outputs = len(smpl_outputs_files)
            num_smpl_verts_cano = len(smpl_verts_cano_files)
            if num_frames != num_masks or num_frames != num_masked_images or num_frames != num_depth_images or num_frames != num_smpl_params or num_frames != num_poses or num_frames != num_intrinsics or num_frames != num_smpl_tfs or num_frames != num_smpl_outputs or num_frames != num_smpl_verts_cano:
                raise ValueError(f"Number of frames ({num_frames}) does not match number of masks ({num_masks}) or the other data ({num_smpl_params}, {num_poses}, {num_intrinsics}, {num_smpl_tfs}, {num_smpl_outputs}, {num_smpl_verts_cano}) for video {video_folder}")

            for i, (frame_file, mask_file, masked_image, depth_image, smpl_params_file, pose_file, intrinsics_file, smpl_tfs_file, smpl_outputs_file, smpl_verts_cano_file) in enumerate(zip(frame_files, mask_files, masked_images, depth_images, smpl_params_files, pose_files, intrinsics_files, smpl_tfs_files, smpl_outputs_files, smpl_verts_cano_files)):
                if i % frame_skip == 0:
                    frame_path = os.path.join(frame_dir, frame_file)
                    mask_path = os.path.join(mask_dir, mask_file)
                    masked_img_path = os.path.join(masked_img_dir, masked_image)
                    depth_img_path = os.path.join(depth_img_dir, depth_image)
                    smpl_params_path = os.path.join(smpl_params_dir, smpl_params_file)
                    pose_path = os.path.join(pose_dir, pose_file)
                    intrinsics_path = os.path.join(intrinsics_dir, intrinsics_file)
                    smpl_tfs_path = os.path.join(smpl_tfs_dir, smpl_tfs_file)
                    smpl_outputs_path = os.path.join(smpl_outputs_dir, smpl_outputs_file)
                    smpl_cano_vers_path = os.path.join(smpl_verts_cano_dir, smpl_verts_cano_file)
                    self.data.append({'frame_path': frame_path, 
                                      'mask_path': mask_path, 
                                      'masked_img_path': masked_img_path, 
                                      'depth_img_path': depth_img_path, 
                                      'smpl_params_path': smpl_params_path, 
                                      'pose_path': pose_path, 
                                      'intrinsics_path': intrinsics_path, 
                                      'smpl_tfs_path': smpl_tfs_path, 
                                      'smpl_outputs_path': smpl_outputs_path,
                                      'smpl_verts_cano_path': smpl_cano_vers_path,
                                      'betas_path': betas_path, 
                                      'metadata': metadata,
                                      'frame_id': i})

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
        # frame = cv2.imread(frame_info['frame_path'])
        ## frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Not sure about this
        # mask = cv2.imread(frame_info['mask_path'], cv2.IMREAD_GRAYSCALE)
        betas = torch.tensor(np.load(frame_info['betas_path'])[None], dtype=torch.float32) 
        metadata = frame_info['metadata']
        
        # masked_image = cv2.bitwise_and(frame, frame, mask=mask)
        # I think we should not resize
        # masked_image = cv2.resize(masked_image, (1080, 1080))   # When reshaping we have to also change the parameters etc. run in the preprocessing of the original image
        masked_image = cv2.imread(frame_info['masked_img_path'], cv2.IMREAD_COLOR) 
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(frame_info['depth_img_path'], cv2.IMREAD_GRAYSCALE)
        original_size = masked_image.shape[:2]
        image_size = self.image_size
        if image_size != 'original':
            if self.crop:
                non_black_indices = np.where(np.any(masked_image != [0, 0, 0], axis=-1))
                min_y = np.min(non_black_indices[0]) - 10
                max_y = np.max(non_black_indices[0]) + 10
                min_x = np.min(non_black_indices[1]) - 10
                max_x = np.max(non_black_indices[1]) + 10
                masked_image = masked_image[min_y:max_y, min_x:max_x]
                depth_image = depth_image[min_y:max_y, min_x:max_x]
                #original_size = masked_image.shape[:2]
            masked_image = cv2.resize(masked_image, image_size)
            depth_image = cv2.resize(depth_image, image_size)
        else:
            image_size = original_size
        masked_image = transforms.ToTensor()(masked_image)
        depth_image = transforms.ToTensor()(depth_image)
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())

        number_of_non_zero = np.count_nonzero(masked_image)
        if number_of_non_zero < 100:   # Check fro degraded images/masks
            raise ValueError(f"Number of non-zero pixels in masked image is {number_of_non_zero}, mask might be degraded")

        #depth_mask = (depth_image > 0.1).float()   # Mask refinement with depth image
        #depth_image = depth_image * depth_mask
        #masked_image = masked_image * depth_mask   

        if self.transform is not None:
            masked_image = self.transform(masked_image)

        '''
        transform = Compose([
                    Resize(
                        width=518,
                        height=518,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method='lower_bound',
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   
                    PrepareForNet(),
                ])
        raw_image_dpt = cv2.imread(frame_info['masked_img_path'])
        image_dpt = cv2.cvtColor(raw_image_dpt, cv2.COLOR_BGR2RGB) / 255.0
        image_dpt = transform({'image': image_dpt})['image']
        image_dpt = torch.from_numpy(image_dpt)

        height, width = image_resnet.shape[1], image_resnet.shape[2]
        pad_height = (32 - height % 32) % 32
        pad_width = (32 - width % 32) % 32

        image_unet_resnet = F.pad(image_resnet, (0, pad_width, 0, pad_height))
        '''
        transform = Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image_resnet = transform(masked_image)

        smpl_params = torch.load(frame_info['smpl_params_path']) # Needed?
        pose = torch.load(frame_info['pose_path']) # Needed? (4, 4)
        intrinsics = torch.load(frame_info['intrinsics_path']) # Needed? (4, 4)
        smpl_tfs = torch.load(frame_info['smpl_tfs_path'])
        smpl_outputs = torch.load(frame_info['smpl_outputs_path'])
        smpl_verts_cano = torch.load(frame_info['smpl_verts_cano_path'])

        inputs = {'masked_image': masked_image, 
                    #'image_dpt': image_dpt,   # Added for depth anything processing
                    'image_resnet': image_resnet,
                    #'image_unet_resnet': image_unet_resnet,
                    'masked_img_path': frame_info['masked_img_path'],   # Added for depth anything processing
                    'depth_image': depth_image,
                    # 'uv': uv,
                    # 'cam_loc': cam_loc,   
                    'original_size': original_size,
                    'image_size': image_size,
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y,
                    'smpl_params': smpl_params, 
                    'pose': pose, 
                    'intrinsics': intrinsics,
                    'smpl_tfs': smpl_tfs,
                    'smpl_outputs': smpl_outputs,
                    'smpl_verts_cano': smpl_verts_cano,
                    'betas': betas,
                    'metadata': metadata,
                    'frame_id': frame_info['frame_id']}

        return inputs
