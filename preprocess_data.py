# 3DPW Dataset Preprocessing:
# Preprocessing the generate files containing SMPL transformations, pose estimation, shape estimation, and camera intrinsics
# After running this script, the initial data will be divided in frame-based files

import argparse
import os
import glob
import cv2
import numpy as np
import torch
import yaml
from lib.utils import utils  
from smpl_server import SMPLServer
from deformer import SMPLDeformer
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import random
import time
import wandb

def preprocess_data(video_folder, process_smpl_output=False, process_depth_img=False, refine_mask=False, check_mask=False, visualize=False):
    wandb.init(project="una-gen-data", config={})
    
    if process_smpl_output:
        print("Processing SMPL output")
        with open(os.path.join(video_folder, "metadata.yaml"), 'r') as file:
            metadata = yaml.safe_load(file)
        gender = metadata['gender']
        training_indices = list(range(metadata['start_frame'], metadata['end_frame']))
        betas = np.load(os.path.join(video_folder, "mean_shape.npy"))

        shape = np.load(os.path.join(video_folder, "mean_shape.npy"))
        poses = np.load(os.path.join(video_folder, 'poses.npy'))[training_indices]
        trans = np.load(os.path.join(video_folder, 'normalize_trans.npy'))[training_indices]
        # cameras
        camera_dict = np.load(os.path.join(video_folder, "cameras_normalize.npz"))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in training_indices]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in training_indices]

        scale = 1 / scale_mats[0][0, 0]

        intrinsics_all = []
        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())
        assert len(intrinsics_all) == len(pose_all)

        smpl_params_folder = os.path.join(video_folder, "smpl_params")
        intrinsics_folder = os.path.join(video_folder, "intrinsics")
        pose_folder = os.path.join(video_folder, "pose")
        os.makedirs(smpl_params_folder, exist_ok=True)
        os.makedirs(intrinsics_folder, exist_ok=True)
        os.makedirs(pose_folder, exist_ok=True)

        for idx in training_indices:
            smpl_params = torch.zeros([86]).float()
            smpl_params[0] = torch.from_numpy(np.asarray(scale)).float() 

            smpl_params[1:4] = torch.from_numpy(trans[idx]).float()
            smpl_params[4:76] = torch.from_numpy(poses[idx]).float()
            smpl_params[76:] = torch.from_numpy(shape).float()

            index_str = "{:04d}".format(idx)
            torch.save(smpl_params, os.path.join(smpl_params_folder, "smpl_params_{}.pt".format(index_str)))
            torch.save(intrinsics_all[idx], os.path.join(intrinsics_folder, "intrinsics_{}.pt".format(index_str)))
            torch.save(pose_all[idx], os.path.join(pose_folder, "pose_{}.pt".format(index_str)))

        smpl_server = SMPLServer(gender=gender, betas=betas)
        # deformer = SMPLDeformer(betas=betas, gender=gender)
        
        smpl_tfs_folder = os.path.join(video_folder, "smpl_tfs")
        smpl_outputs_folder = os.path.join(video_folder, "smpl_outputs")
        smpl_verts_cano_folder = os.path.join(video_folder, "smpl_verts_cano")
        os.makedirs(smpl_tfs_folder, exist_ok=True)
        os.makedirs(smpl_outputs_folder, exist_ok=True)
        os.makedirs(smpl_verts_cano_folder, exist_ok=True)
        
        smpl_params_files = sorted([f for f in os.listdir(smpl_params_folder) if f.endswith(".pt")])
        for i, filename in enumerate(smpl_params_files):
            print("Processing file", i, "of", len(smpl_params_files), end="\r")
            smpl_params_path = os.path.join(smpl_params_folder, filename)
            smpl_params = torch.load(smpl_params_path)
            smpl_params = smpl_params.unsqueeze(0)

            scale, smpl_trans, smpl_pose, smpl_shape = torch.split(smpl_params, [1, 3, 72, 10], dim=1)
            scale = scale.cuda()
            smpl_trans = smpl_trans.cuda()
            smpl_pose = smpl_pose.cuda()
            smpl_shape = smpl_shape.cuda()
            
            smpl_outputs = smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)

            smpl_vertices = smpl_outputs['smpl_verts']
            smpl_tfs = smpl_outputs['smpl_tfs']

            betas_path = os.path.join(video_folder, "mean_shape.npy")
            betas = torch.tensor(np.load(betas_path)[None], dtype=torch.float32)
            metadata_path = os.path.join(video_folder, "metadata.yaml")

            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)

            deformer = SMPLDeformer(betas=betas, gender=gender)
            smpl_verts_cano, outlier_mask = deformer.forward(smpl_vertices[0], smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_vertices)

            smpl_verts_cano = smpl_verts_cano.cpu()
            smpl_tfs = smpl_tfs.cpu()
            
            torch.save(smpl_tfs, os.path.join(smpl_tfs_folder, filename))
            torch.save(smpl_outputs, os.path.join(smpl_outputs_folder, filename))
            torch.save(smpl_verts_cano, os.path.join(smpl_verts_cano_folder, filename))

    if process_depth_img:
        print("Processing Depth Image")
        rgb_frames_folder = os.path.join(video_folder, "rgb_frames")
        masks_folder = os.path.join(video_folder, "masks")
        refined_mask_folder = os.path.join(video_folder, "refined_masks")
        checked_mask_folder = os.path.join(video_folder, "checked_masks")
        masked_img_folder = os.path.join(video_folder, "masked_img")
        os.makedirs(masked_img_folder, exist_ok=True)
        depth_img_folder = os.path.join(video_folder, "depth_img_bw")
        os.makedirs(depth_img_folder, exist_ok=True)

        time_taken = 0
        for i, filename in enumerate(os.listdir(rgb_frames_folder)):
            print("ETA:", round((len(os.listdir(rgb_frames_folder))-i)*time_taken/60), "m", "|" , round(i/len(os.listdir(rgb_frames_folder))*100), "%", "|", "time per mask:", round(time_taken), "s")
            t0 = time.time()
            rgb_frame = cv2.imread(os.path.join(rgb_frames_folder, filename))
            #rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            if os.path.exists(checked_mask_folder):
                mask = cv2.imread(os.path.join(checked_mask_folder, filename), cv2.IMREAD_GRAYSCALE)
                wandb.log({"Mask": [wandb.Image(mask)]})
            elif os.path.exists(refined_mask_folder):
                mask = cv2.imread(os.path.join(refined_mask_folder, filename), cv2.IMREAD_GRAYSCALE)
                wandb.log({"Mask": [wandb.Image(mask)]})
            elif os.path.exists(masks_folder):
                mask = cv2.imread(os.path.join(masks_folder, filename), cv2.IMREAD_GRAYSCALE)
                wandb.log({"Mask": [wandb.Image(mask)]})
            else:
                mask = None

            if mask is not None:
                masked_image = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)   # Apply mask to remove background
            else:
                masked_image = rgb_frame
            cv2.imwrite(os.path.join(masked_img_folder, filename), masked_image)   # Save masked image
            command = ["python3", "run.py", "--encoder", "vitl", "--img-path", os.path.join(masked_img_folder, filename), "--outdir", depth_img_folder, "--pred-only", "--grayscale"]
            subprocess.run(command, check=True, cwd="/UnA-Gen/supp_repos/Depth_Anything_main")
            filename_without_extension = filename.rsplit('.', 1)[0]  # Split on the last '.' and take the first part
            filename = f"{filename_without_extension}_depth.png"
            depth_image = cv2.imread(os.path.join(depth_img_folder, filename), cv2.IMREAD_COLOR) 
            wandb.log({"Depth Image": [wandb.Image(depth_image)]})
            if mask is not None:
                depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
            else:
                depth_image = depth_image
            wandb.log({"Depth Image after mask": [wandb.Image(depth_image)]})
            cv2.imwrite(os.path.join(depth_img_folder, filename), depth_image)   # Apply mask again to remove background noise

            t1 = time.time()
            time_taken = t1 - t0

    if refine_mask:
        print("Refining Masks")
        rgb_frames_folder = os.path.join(video_folder, "rgb_frames")
        masks_folder = os.path.join(video_folder, "masks")
        masked_images_folder = os.path.join(video_folder, "masked_img")
        refined_mask_folder = os.path.join(video_folder, "refined_masks")
        os.makedirs(refined_mask_folder, exist_ok=True)

        sam_checkpoint = "/UnA-Gen/supp_repos/Segment_Anything/checkpoint/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device="cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)
        #predictor = SamPredictor(sam)

        time_taken = 0
        for i, filename in enumerate(os.listdir(rgb_frames_folder)):
            print("ETA:", round((len(os.listdir(rgb_frames_folder))-i)*time_taken/60), "m", "|" , round(i/len(os.listdir(rgb_frames_folder))*100), "%", "|", "time per mask:", round(time_taken), "s", end="\r")
            t0 = time.time()
            rgb_frame = cv2.imread(os.path.join(rgb_frames_folder, filename))
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(masks_folder, filename), cv2.IMREAD_GRAYSCALE).astype(np.uint8)

            #predictor.set_image(rgb_frame)
            #output_list, scores, logits = predictor.predict()
            
            output_list = mask_generator.generate(rgb_frame)

            max_int = 0
            for i, output in enumerate(output_list):
                intersection_with_mask = np.logical_and(output['segmentation'], mask)
                #intersection_with_mask = np.logical_and(output, mask)
                number_of_intersected_pixels = np.count_nonzero(intersection_with_mask)
                if number_of_intersected_pixels > max_int:
                    max_int = number_of_intersected_pixels
                    max_int_index = i            

            #print("masks.shape: ", masks.shape)
            #print("scores.shape: ", scores.shape)
            #print("masks: ", masks)
            #print("scores: ", scores)
            #print("number of non zero elements in mask: ", np.count_nonzero(masks[0]))
            #mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            #print("mask_input:", mask_input.shape)
            #print("mask_input: ", mask_input)

            #refined_mask = predictor.predict(rgb_frame, mask)
            cv2.imwrite(os.path.join(refined_mask_folder, filename), (output_list[max_int_index]['segmentation'] * 255).astype(np.uint8))
            #cv2.imwrite(os.path.join(refined_mask_folder, filename), (output_list[max_int_index] * 255).astype(np.uint8))

            t1 = time.time()
            time_taken = t1 - t0

    if check_mask:
        print("Checking Masks")
        rgb_frames_folder = os.path.join(video_folder, "rgb_frames")
        masks_folder = os.path.join(video_folder, "masks")
        refined_mask_folder = os.path.join(video_folder, "refined_masks")
        checked_mask_folder = os.path.join(video_folder, "checked_masks")
        os.makedirs(checked_mask_folder, exist_ok=True)

        number_of_failed_refinements = 0
        number_of_successful_openings = 0
        number_of_successful_closings = 0
        time_taken = 0
        for i, filename in enumerate(os.listdir(rgb_frames_folder)):
            print("ETA:", round((len(os.listdir(rgb_frames_folder))-i)*time_taken/60), "m", "|" , round(i/len(os.listdir(rgb_frames_folder))*100), "%", "|", "time per mask:", round(time_taken), "s", end="\r")
            t0 = time.time()
            mask = cv2.imread(os.path.join(masks_folder, filename), cv2.IMREAD_GRAYSCALE)
            refined_mask = cv2.imread(os.path.join(refined_mask_folder, filename), cv2.IMREAD_GRAYSCALE)
            intersection_with_mask = np.logical_and(refined_mask, mask)
            n_int_pixels = np.count_nonzero(intersection_with_mask)
            iou = n_int_pixels / np.count_nonzero(mask)
            if iou < 0.8:
                number_of_failed_refinements += 1
                checked_mask = mask
                wandb.log({"Failed Refinements: refined_mask": [wandb.Image(refined_mask)]})
                wandb.log({"Failed Refinements: mask": [wandb.Image(mask)]})
            else:
                checked_mask = refined_mask

            kernel = np.ones((5,5), np.uint8)  # You can adjust the kernel size as needed
            checked_mask = cv2.erode(checked_mask, kernel, iterations=1)
            checked_mask = cv2.dilate(checked_mask, kernel, iterations=1)     
            new_n_int_pixels = np.count_nonzero(np.logical_and(checked_mask, mask))
            if new_n_int_pixels < n_int_pixels:
                number_of_successful_openings += 1      
                n_int_pixels

            kernel = np.ones((5,5), np.uint8)  # You can adjust the kernel size as needed
            checked_mask = cv2.dilate(checked_mask, kernel, iterations=1)
            checked_mask = cv2.erode(checked_mask, kernel, iterations=1)
            new_n_int_pixels = np.count_nonzero(np.logical_and(checked_mask, mask))
            if new_n_int_pixels < n_int_pixels:
                number_of_successful_closings += 1
            
            cv2.imwrite(os.path.join(checked_mask_folder, filename), checked_mask)

            t1 = time.time()
            time_taken = t1 - t0
        
        print("Number of failed refinements:", number_of_failed_refinements)
        print("Number of successful openings:", number_of_successful_openings)
        print("Number of successful closings:", number_of_successful_closings)

    return


if __name__ == "__main__":
    video_folder = '/UnA-Gen/data/data/train/0025_11'
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, default=video_folder)
    parser.add_argument('--smpl_output', action='store_true', help='Use SMPL server to obtain SMPL outputs')
    parser.add_argument('--depth_img', action='store_true', help='Use Depth Anything to obtain depth image')
    parser.add_argument('--refine_mask', action='store_true', help='Use SAM to refine mask')
    parser.add_argument('--check_mask', action='store_true', help='Check if the masks are correct')
    args = parser.parse_args()
    preprocess_data(args.video_folder, args.smpl_output, args.depth_img, args.refine_mask, args.check_mask)
    