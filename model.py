import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DoubleConv, DownSample, UpSample
from deformer import SMPLDeformer, skinning
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 32)
        self.down_convolution_2 = DownSample(32, 64)
        self.down_convolution_3 = DownSample(64, 128)
        #self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(128, 256)

        #self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(256, 128)
        self.up_convolution_3 = UpSample(128, 64)
        self.up_convolution_4 = UpSample(64, 32)
        
        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        #down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p3)

        #up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(b, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out

class OccupancyField(nn.Module):
    def __init__(self, resolution=256, in_features=131, num_classes=1):
        super(OccupancyField, self).__init__()
        self.resolution = resolution
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Linear(in_features, 131),
            nn.ReLU(inplace=True),
            nn.Linear(131, 131),
            nn.ReLU(inplace=True),
            nn.Linear(131, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class RGBfield(nn.Module):
    def __init__(self, resolution=256, in_features=131, num_classes=3):
        super(RGBfield, self).__init__()
        self.resolution = resolution
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Linear(in_features, 131),
            nn.ReLU(inplace=True),
            nn.Linear(131, 131),
            nn.ReLU(inplace=True),
            nn.Linear(131, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
# Main model: UnA-Gen -------------------------------------------------------------------------------------------------

class UnaGenModel(nn.Module):
    def __init__(self, in_channels=3, features=128, mmap_dim=1024, mmap_res=256, num_classes=1):
        super(UnaGenModel, self).__init__()
        self.GeometryEncoder = UNetEncoder(in_channels=in_channels, out_channels=features).cuda()
        self.RenderingEncoder = UNetEncoder(in_channels=in_channels, out_channels=features).cuda()

        self.mmap_dim = mmap_dim
        self.mmap_res = mmap_res
        self.num_classes = num_classes

        self.OccupancyField = OccupancyField(resolution=mmap_res, in_features=features+3, num_classes=num_classes).cuda()
        self.RGBField = RGBfield(resolution=mmap_res, in_features=features+3, num_classes=3).cuda()

        self.debug = True

    def forward(self, inputs, matrix_mapping):
        # input is a dictionary containing the image, the smpl parameters, and other info 
        # it is of shape (batch_size, key_values) where key values is of different dimensions
        
        image = inputs['masked_image']

        gender = inputs['metadata']['gender']
        betas = inputs['betas']
        # scale = inputs['smpl_params'][:, 0].cuda()
        # smpl_pose = inputs["pose"].cuda()
        # intrinsics = inputs['intrinsics'].cuda()
        smpl_tfs = inputs['smpl_tfs'].cuda()

        with torch.no_grad():
            # The active voxels are mapped by the matrix_mapping (while in the canonical space)
            # TODO: Implement the mapping of the active voxels by the matrix_mapping 
            active_voxels_coo = self.voxel_mapping(matrix_mapping)   # Returns the active voxels coordinates in the canonical space
            self.visualize_voxels(active_voxels_coo, output_file='active_voxels.png')

            # The active voxels are transformed to the dynamical space from the canonical space using the smpl pose parameters
            # TODO: Implement the transformation of the active voxels to the dynamical space using the smpl pose parameters
            # dynamical_voxels_coo = transform_voxels(active_voxels_coo, smpl_tfs)   # Returns the active voxels coordinates in the dynamical space

            dynamical_voxels_coo = self.dynamical_tfs_from_canonical(active_voxels_coo, smpl_tfs, betas, gender)   # To be tested if it works
            self.visualize_voxels(dynamical_voxels_coo[0], output_file='dynamical_voxels.png')

        # ---
        # The feature encoder produces a pixel/voxel-wise feature embedding used to compute the occupancy field of the active voxels
        features = self.GeometryEncoder(image)   # (batch_size, features, img_width, img_height)

        # The feature embedding is interpolated to the active voxels in the dynamical space
        # TODO: Implement the interpolation of the feature embedding to the active voxels in the dynamical space (coordinates to pixel/voxel aligned features)
        features_at_voxels = self.interpolate_features(features, dynamical_voxels_coo)

        # The occupancy fields infers all the active voxels in the dynamical space  using the feature embedding
        # TODO: implement to run faster       
        occupancy_field = self.OccupancyField(features_at_voxels.view(-1, features_at_voxels.shape[-1]))
        occupancy_field = occupancy_field.view(features_at_voxels.shape[0], features_at_voxels.shape[1], -1)

        # --- this is currently not used, as the occupancy field is used to produce the mask during the rendering
        # Apply a threshold to the occupancy field
        # occupied_voxels_mask = occupancy_field > 0.5
        # occupied_voxels = dynamical_voxels_coo[occupied_voxels_mask.squeeze(-1)]

        # --- Rendering ---
        features_r = self.RenderingEncoder(image)

        features_r_at_voxels = self.interpolate_features(features_r, dynamical_voxels_coo)

        rgb_field = self.RGBField(features_r_at_voxels.view(-1, features_r_at_voxels.shape[-1]))
        rgb_field = rgb_field.view(features_r_at_voxels.shape[0], features_r_at_voxels.shape[1], -1)
        # ---

        # The output of the occupancy field is used to compute the volume if needed
        # TODO: Implement the computation of the volume using the occupancy field as a separate method of the model, which can be called from outside

        # Also, the output of the occupancy field is used to refine the active voxels by modifying the matrix_mapping
        # The matrix mapping is to be kept larger than the actually occupied voxels, to allow for garment details etc.
        # However, where the active voxels are repeatedly not occupied, the matrix_mapping is to be reduced
        # The resolution (number of active voxels) reduces, while the number of occupied voxels remains the same (propotionally)
        # When the resolution is reduced enough, the matrix_mapping is to be refined to smaller voxels
        # TODO: implement the refinement of the matrix_mapping using the occupancy field as a separate method of the model, which can be called from outside
        # TODO: alternatively, make the matrix_mapping a property of the model, and check if refinement is due in the forward method
        
        # render image
        rendered_image = self.render_image(dynamical_voxels_coo, occupancy_field, rgb_field, image)

        outputs = {'dynamical_voxels_coo': dynamical_voxels_coo, 
                   'occupancy_field': occupancy_field, 
                   'rgb_field': rgb_field,
                   'rendered_image': rendered_image}

        return outputs
    
    # Auxiliary methods ---------------------------------------------------------------------------------------------------
    def visualize_voxels(self, voxels_coo, output_file=None):
        '''This is used to save an image of the voxels in 3D space.
        '''
        voxels_coo_np = voxels_coo.detach().cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(voxels_coo_np[:, 0], voxels_coo_np[:, 1], voxels_coo_np[:, 2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if output_file is None:
            output_file = 'voxels_temp.png'
        output_file = os.path.join('outputs/debug/', output_file)
        plt.savefig(output_file)

        return


    def interpolate_features(self, features, voxels_coo):
        '''At the moment we simply interpolate the features using the nearest neighbor interpolation.
        Later on, we'll have to change it to a more sophisticated interpolation method.
        N.B.: The coordinates are normalized in the min-max range, because the skinning transformation is deforming the [-1,1] range. (we need to check if this is correct)
        '''
        height, width = features.shape[2], features.shape[3]
        y = voxels_coo[:, :, 0]
        x = voxels_coo[:, :, 1]
        y_min, y_max = torch.min(y), torch.max(y)
        x_min, x_max = torch.min(x), torch.max(x)
        fy = (y - y_min) / (y_max - y_min) * (height - 1)
        fx = (x - x_min) / (x_max - x_min) * (width - 1)
        fy = fy.long()
        fx = fx.long()

        interpolated_features = torch.zeros(features.size(0), voxels_coo.size(1), features.size(1) + voxels_coo.size(2)).to(features.device)

        batch_indices = torch.arange(features.size(0)).unsqueeze(1).expand_as(fy).to(features.device)

        interpolated_features[:, :, :features.shape[1]] = features[batch_indices, :, fy, fx]
        interpolated_features[:, :, features.shape[1]:] = voxels_coo
        
        return interpolated_features
    
    def voxel_mapping(self, matrix_mapping):
        active_voxels_coo = torch.nonzero(matrix_mapping == 1).float()

        for i in range(active_voxels_coo.shape[-1]):
            dim_min = 0
            dim_max = matrix_mapping.shape[i] - 1
            active_voxels_coo[:, i] = 2 * (active_voxels_coo[:, i] - dim_min) / (dim_max - dim_min) - 1

        return active_voxels_coo
    
    def dynamical_tfs_from_canonical(self, points_coo, smpl_tfs, betas, gender):
        verts_deformed = torch.zeros(len(gender), points_coo.shape[0], points_coo.shape[1]).cuda().float()
        for i, gen in enumerate(gender):   # Each frame in the batch needs to have the skinning weights loaded separately
            deformer = SMPLDeformer(betas=betas[i], gender=gen)
            weights = deformer.query_weights(points_coo)
            verts_deformed[i] = skinning(points_coo.unsqueeze(0), weights, smpl_tfs[i]).data[0]
        return verts_deformed
    
    def render_image(self, dynamical_voxels_coo, occupancy_field, rgb_field, original_image):
        '''Render image from occupancy field and rgb field.'''
        height, width = original_image.shape[2:]

        occupancy_map = (occupancy_field > -1).float() # TODO: change back to 0.5
        occupied_voxels_coo = dynamical_voxels_coo * occupancy_map # Should change this, puts 0s in the empty voxels coordinates
        occupied_voxels_rgb = rgb_field * occupancy_map

        rendered_image = torch.zeros(original_image.shape, device=original_image.device)
        
        for image in range(original_image.shape[0]):
            max_x = occupied_voxels_coo[:,:,0].max()
            max_y = occupied_voxels_coo[:,:,1].max()
            min_x = occupied_voxels_coo[:,:,0].min()
            min_y = occupied_voxels_coo[:,:,1].min()
            for voxel in range(occupied_voxels_coo.shape[1]):
                x = occupied_voxels_coo[image, voxel, 0]
                y = occupied_voxels_coo[image, voxel, 1]
                ix = (x - min_x)/(max_x - min_x) * (width - 1)
                iy = (y - min_y)/(max_y - min_y) * (height - 1)
                ix = ix.long()
                iy = iy.long()
                rendered_image[image, :, iy, ix] =+ occupied_voxels_rgb[image, voxel, :]
        
        rendered_image = rendered_image*255

        return rendered_image