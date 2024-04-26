import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DoubleConv, DownSample, UpSample
from deformer import SMPLDeformer, skinning
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from torchvision.utils import make_grid, save_image
import time
from utils import quat_to_rot

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 16)
        self.down_convolution_2 = DownSample(16, 32)
        self.down_convolution_3 = DownSample(32, 64)
        #self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(64, 128)

        #self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(128, 64)
        self.up_convolution_3 = UpSample(64, 32)
        self.up_convolution_4 = UpSample(32, 16)
        
        self.out = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1)

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
    def __init__(self, in_features=131, num_classes=1):
        super(OccupancyField, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 131),
            nn.ReLU(inplace=True),
            nn.Linear(131, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class RGBfield(nn.Module):
    def __init__(self, in_features=131, num_classes=3):
        super(RGBfield, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 131),
            nn.ReLU(inplace=True),
            nn.Linear(131, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
# Main model: UnA-Gen -------------------------------------------------------------------------------------------------
class UnaGenModel(nn.Module):
    def __init__(self, opt, in_channels=3, features=128):
        super(UnaGenModel, self).__init__()
        
        self.n_training_rays = opt['n_training_rays']
        self.mapping_dim = opt['mapping_dim']
        self.matrix_mapping = self.initialize_matrix_mapping(self.mapping_dim, 0.5).cuda()
        self.visualize_stats = opt['visualize_stats']
        if self.visualize_stats:
            self.time_forward_pass = []
            self.time_rgb_values_rendering = []

        self.GeometryEncoder = UNetEncoder(in_channels=in_channels, out_channels=features).cuda()
        self.RenderingEncoder = UNetEncoder(in_channels=in_channels, out_channels=features).cuda()

        '''
        for param in self.GeometryEncoder.parameters():
            param.requires_grad = False

        for param in self.RenderingEncoder.parameters():
            param.requires_grad = False
        '''

        self.OccupancyField = OccupancyField(in_features=features+3, num_classes=1).cuda()
        self.RGBField = RGBfield(in_features=features+3, num_classes=3).cuda()

        '''
        for param in self.OccupancyField.parameters():
            param.requires_grad = False

        for param in self.RGBField.parameters():
            param.requires_grad = False
        '''

        self.apply(self.weights_init)   # weights initialization

    def forward(self, inputs):
        # input is a dictionary containing the image, the smpl parameters, and other info 
        # it is of shape (batch_size, key_values) where key values is of different dimensions
        t0 = time.time()        
        image = inputs['masked_image']

        gender = inputs['metadata']['gender']
        betas = inputs['betas']
        # scale = inputs['smpl_params'][:, 0].cuda()
        # pose = inputs["pose"].cuda()
        # intrinsics = inputs['intrinsics'].cuda()
        smpl_tfs = inputs['smpl_tfs'].cuda()

        matrix_mapping = self.matrix_mapping

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
        features = self.interpolate_features(features, dynamical_voxels_coo)

        # The occupancy fields infers all the active voxels in the dynamical space  using the feature embedding
        # TODO: implement to run faster       
        occupancy_field = self.OccupancyField(features.view(-1, features.shape[-1]))
        occupancy_field = occupancy_field.view(features.shape[0], features.shape[1], -1)

        # --- this is currently not used, as the occupancy field is used to produce the mask during the rendering
        # Apply a threshold to the occupancy field
        # occupied_voxels_mask = occupancy_field > 0.5
        # occupied_voxels = dynamical_voxels_coo[occupied_voxels_mask.squeeze(-1)]

        # --- Rendering ---
        features_r = self.RenderingEncoder(image)

        features_r = self.interpolate_features(features_r, dynamical_voxels_coo)

        rgb_field = self.RGBField(features_r.view(-1, features_r.shape[-1]))
        rgb_field = rgb_field.view(features_r.shape[0], features_r.shape[1], -1)
        # ---

        # The output of the occupancy field is used to compute the volume if needed
        # TODO: Implement the computation of the volume using the occupancy field as a separate method of the model, which can be called from outside

        # Also, the output of the occupancy field is used to refine the active voxels by modifying the matrix_mapping
        # The matrix mapping is to be kept larger than the actually occupied voxels, to allow for garment details etc.
        # However, where the active voxels are repeatedly not occupied, the matrix_mapping is to be reduced
        # The resolution (number of active voxels) reduces, while the number of occupied voxels remains the same (propotionally)
        # When the resolution is reduced enough, the matrix_mapping is to be refined to smaller voxels
        # TODO: implement the refinement of the matrix_mapping using the occupancy field as a separate method of the model, which can be called from outside
        # matrix_mapping = self.update_matrix_mapping(matrix_mapping, occupancy_field)
        # This update may be better done in the train script
        
        # render image
        #dense_occupancy_field = self.voxel_coo_to_dense(dynamical_voxels_coo, occupancy_field, rgb_field)

        t0_render_rgb_values = time.time()
        original_rgb_values, rendered_rgb_values = self.render_values_at_rays(dynamical_voxels_coo, occupancy_field, rgb_field, image)
        t1_render_rgb_values = time.time()

        if self.visualize_stats:
            self.time_rgb_values_rendering.append(t1_render_rgb_values - t0_render_rgb_values)
            plt.plot(self.time_rgb_values_rendering)
            plt.axhline(np.mean(self.time_rgb_values_rendering), color='red')
            plt.title(f'RGB values rendering, avg: {np.mean(self.time_rgb_values_rendering):.4f}')
            plt.ylabel('Time')
            plt.xlabel('Batch')
            plt.savefig('outputs/stats/RGB_values_rendering.png')
            plt.close()

        '''
        print("occupancy_field max:", occupancy_field.max().item())
        print("occupancy_field min:", occupancy_field.min().item())
        print("number of occupied voxels:", torch.count_nonzero(occupancy_field > 0.5).item())
        print("rgb_field max:", rgb_field.max().item())
        print("rgb_field min:", rgb_field.min().item())
        print("rendered_rgb_values max:", rendered_rgb_values.max().item())
        print("rendered_rgb_values min:", rendered_rgb_values.min().item())
        '''

        outputs = {'dynamical_voxels_coo': dynamical_voxels_coo, 
                   'occupancy_field': occupancy_field, 
                   'rgb_field': rgb_field,
                   'original_rgb_values': original_rgb_values, 
                   'rendered_rgb_values': rendered_rgb_values}

        t1 = time.time()
        if self.visualize_stats:
            self.time_forward_pass.append(t1-t0)
            plt.plot(self.time_forward_pass)
            plt.axhline(np.mean(self.time_forward_pass), color='red')
            plt.title(f'Forward pass, avg: {np.mean(self.time_forward_pass):.4f}')
            plt.ylabel('Time')
            plt.xlabel('Batch')
            plt.savefig('outputs/stats/forward_pass_time.png')
            plt.close()
        
        if False:
            ren_t0 = time.time()
            original_image, rendered_image = self.render_image(dynamical_voxels_coo, occupancy_field, rgb_field, image)
            ren_t1 = time.time()
            print(f'Rendering took {ren_t1-ren_t0} seconds')
            original_image = original_image.view(2, 3, 1920, 1080)
            rendered_image = rendered_image.view(2, 3, 1920, 1080)
            save_image(original_image, 'outputs/stats/original_image.png')
            save_image(rendered_image, 'outputs/stats/rendered_image.png')
        
        return outputs
    
    # Auxiliary methods ---------------------------------------------------------------------------------------------------
    def initialize_matrix_mapping(self, mmap_dim, mmap_res):
        '''This method is used to initialize the matrix_mapping with random positions close to the center.
        '''
        matrix_mapping = torch.ones(mmap_dim, mmap_dim, mmap_dim)
        # TODO: implement the random selection of inner voxels

        return matrix_mapping
    
    def update_matrix_mapping(self, matrix_mapping, occupancy_field):
        '''This method is used to update the matrix_mapping based on the occupancy field.
        '''
        return matrix_mapping

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
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        if output_file is None:
            output_file = 'voxels_temp.png'
        output_file = os.path.join('outputs/stats/', output_file)
        plt.savefig(output_file)
        plt.close()

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
    
    def voxel_coo_to_dense(self, dynamical_voxels_coo, occupancy_field, rgb_field):
        '''This method is used to convert the dynamical voxels coordinates to a dense representation.
        '''
        mapping_dim = self.mapping_dim
        dense_occupancy_field = torch.zeros(dynamical_voxels_coo.shape[0], mapping_dim, mapping_dim, mapping_dim, rgb_field.shape[2]+1).cuda()
        
        occupancy_map = (occupancy_field > 0.5).float() # TODO: change back to 0.5
        if torch.count_nonzero(occupancy_map)==0:
            occupancy_map[:, 0, 0] = 1  # This is to avoid the case where there are no occupied voxels
        occupied_voxels_coo = dynamical_voxels_coo * occupancy_map # Should change this, puts 0s in the empty voxels coordinates
        occupied_voxels_ov = occupancy_field * occupancy_map
        occupied_voxels_rgb = rgb_field * occupancy_map
        
        for sample in range(occupied_voxels_coo.shape[0]):
            for idx, coo in enumerate(occupied_voxels_coo[sample]):
                x, y, z = coo
                x = int((min(x,1) + 1) / 2 * (mapping_dim - 1))
                y = int((min(y,1) + 1) / 2 * (mapping_dim - 1))
                z = int((min(z,1) + 1) / 2 * (mapping_dim - 1))
                dense_occupancy_field[sample, x, y, z, 0] = occupied_voxels_ov[sample, idx]
                dense_occupancy_field[sample, x, y, z, 1:] = occupied_voxels_rgb[sample, idx]

        return dense_occupancy_field
    
    def dynamical_tfs_from_canonical(self, points_coo, smpl_tfs, betas, gender):
        verts_deformed = torch.zeros(len(gender), points_coo.shape[0], points_coo.shape[1]).cuda().float()
        for i, gen in enumerate(gender):   # Each frame in the batch needs to have the skinning weights loaded separately
            deformer = SMPLDeformer(betas=betas[i], gender=gen)
            weights = deformer.query_weights(points_coo)
            verts_deformed[i] = skinning(points_coo.unsqueeze(0), weights, smpl_tfs[i]).data[0]
        return verts_deformed
    
    def render_values_at_rays(self, dynamical_voxels_coo, occupancy_field, rgb_field, original_image):
        '''Render image from occupancy field and rgb field.'''
        height, width = original_image.shape[2:]
        n_training_rays = torch.tensor(self.n_training_rays).cuda()

        original_rgb_values = torch.zeros(original_image.shape[0], n_training_rays, 3, device=original_image.device)
        rendered_rgb_values = torch.zeros(original_image.shape[0], n_training_rays, 3, device=original_image.device)
        
        random_height_indices = torch.randint(0, height, (n_training_rays,))
        random_width_indices = torch.randint(0, width, (n_training_rays,))
        selected_indices = torch.stack((random_height_indices, random_width_indices), dim=1)
        original_rgb_values[:] = original_image[:, :, selected_indices[:, 0], selected_indices[:, 1]].permute(0, 2, 1)

        occupancy_map = (occupancy_field > 0.5).float() # TODO: change back to 0.5
        if torch.count_nonzero(occupancy_map)==0:
            occupancy_map[:, 0, 0] = 1  # This is to avoid the case where there are no occupied voxels
        occupied_voxels_coo = dynamical_voxels_coo * occupancy_map # Should change this, puts 0s in the empty voxels coordinates
        occupied_voxels_ov = occupancy_field * occupancy_map
        occupied_voxels_rgb = rgb_field * occupancy_map

        if self.visualize_stats:
            self.visualize_voxels(occupied_voxels_coo[0], output_file='occupied_voxels.png')
            occupied_canonical_voxels = occupancy_map[0].view(self.matrix_mapping.shape)
            occupied_canonical_voxels_coo = self.voxel_mapping(occupied_canonical_voxels)
            self.visualize_voxels(occupied_canonical_voxels_coo, output_file='occupied_canonical_voxels.png')
        
        for n, idx in enumerate(selected_indices):
            j = idx[0]
            i = idx[1]
            y = (j / (height - 1)) * 2 - 1
            x = (i / (width - 1)) * 2 - 1
                          
            # distances = torch.sqrt((occupied_voxels_coo[:, :, 0] - y)**2 + (occupied_voxels_coo[:, :, 1] - x)**2 + occupied_voxels_coo[:, :, 2]**2).cuda()
            distances = ((occupied_voxels_coo[:, :, 0] - y)**2 + (occupied_voxels_coo[:, :, 1] - x)**2 + occupied_voxels_coo[:, :, 2]**2).cuda()

            rendered_rgb_values[:, n, :] = torch.sum(
                                                    # occupied_voxels_ov[:, :, :].repeat(1, 1, 3) * 
                                                    occupied_voxels_rgb[:, :, :] * 
                                                    (1 - torch.exp(-distances)).unsqueeze(-1).repeat(1, 1, 3), dim=1)/torch.sum(1 - torch.exp(-distances), dim=1).unsqueeze(-1)

        if self.visualize_stats:
            sqrt_n_training_rays = int(torch.sqrt(n_training_rays).item())
            if sqrt_n_training_rays % 1 != 0:
                raise ValueError("The number of training rays is not a perfect square. Cannot be visualized correctly")
            rgb_values_differences = original_rgb_values - rendered_rgb_values
            original_values = original_rgb_values.view(original_image.shape[0], 3, sqrt_n_training_rays, sqrt_n_training_rays)
            rendered_values = rendered_rgb_values.view(original_image.shape[0], 3, sqrt_n_training_rays, sqrt_n_training_rays)
            differences_values = rgb_values_differences.view(original_image.shape[0], 3, sqrt_n_training_rays, sqrt_n_training_rays)
            images = torch.cat((original_values, rendered_values, differences_values))
            grid = make_grid(images, nrow=4)  # Arrange the images in a 3x4 grid
            save_image(grid, 'outputs/stats/original_vs_rendered_rgb_values.png')
        
        # save_image(torch.cat((original_rgb_values.view(original_image.shape[0], 3, sqrt_n_training_rays, sqrt_n_training_rays), rendered_rgb_values.view(original_image.shape[0], 3, sqrt_n_training_rays, sqrt_n_training_rays))), 'outputs/stats/original_vs_rendered_rgb_values.png')
        
        return original_rgb_values, rendered_rgb_values
    
    
    def render_image(self, dynamical_voxels_coo, occupancy_field, rgb_field,  original_image):
        '''Render image from occupancy field and rgb field.'''
        height, width = original_image.shape[2:]

        rendered_rgb_values = torch.zeros(original_image.shape[0], height*width, 3, device=original_image.device)
        
        height_indices = torch.arange(height)
        width_indices = torch.arange(width)
        grid_height_indices, grid_width_indices = torch.meshgrid(height_indices, width_indices)
        selected_indices = torch.stack((grid_height_indices.flatten(), grid_width_indices.flatten()), dim=1)

        occupancy_map = (occupancy_field > 0.5).float() # TODO: change back to 0.5
        if torch.count_nonzero(occupancy_map)==0:
            occupancy_map[:, 0, 0] = 1  # This is to avoid the case where there are no occupied voxels
        occupied_voxels_coo = dynamical_voxels_coo * occupancy_map # Should change this, puts 0s in the empty voxels coordinates
        occupied_voxels_ov = occupancy_field * occupancy_map
        occupied_voxels_rgb = rgb_field * occupancy_map

        for n, idx in enumerate(selected_indices):
            j = idx[0]
            i = idx[1]
            y = (j / (height - 1)) * 2 - 1
            x = (i / (width - 1)) * 2 - 1
                          
            # distances = torch.sqrt((occupied_voxels_coo[:, :, 0] - y)**2 + (occupied_voxels_coo[:, :, 1] - x)**2 + occupied_voxels_coo[:, :, 2]**2).cuda()
            distances = ((occupied_voxels_coo[:, :, 0] - y)**2 + (occupied_voxels_coo[:, :, 1] - x)**2 + occupied_voxels_coo[:, :, 2]**2).cuda()

            rendered_rgb_values[:, n, :] = torch.sum(
                                                    # occupied_voxels_ov[:, :, :].repeat(1, 1, 3) * 
                                                    occupied_voxels_rgb[:, :, :] * 
                                                    (1 - torch.exp(-distances)).unsqueeze(-1).repeat(1, 1, 3), dim=1)/torch.sum(1 - torch.exp(-distances), dim=1).unsqueeze(-1)

        return rendered_rgb_values
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
