import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DoubleConv, DownSample, UpSample
from deformer import SMPLDeformer, skinning
from smpl_server import SMPLServer

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
    def __init__(self, resolution=256, in_features=128, num_classes=1):
        super(OccupancyField, self).__init__()
        self.resolution = resolution
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Reshape the input tensor to (batch_size, in_features * resolution)
        x = x.view(-1, self.in_features * self.resolution)
        x = self.fc(x)
        return x
    
# Main model: UnA-Gen -------------------------------------------------------------------------------------------------

class UnaGenModel(nn.Module):
    def __init__(self, in_channels=3, features=128, mmap_dim=1024, mmap_res=256, num_classes=1):
        super(UnaGenModel, self).__init__()
        self.feature_encoder = UNetEncoder(in_channels=in_channels, out_channels=features)
        self.mmap_dim = mmap_dim
        self.mmap_res = mmap_res
        self.num_classes = num_classes
        self.classifier = OccupancyField(resolution=mmap_res, in_features=features, num_classes=num_classes)

        self.smpl_server = None
        self.deformer = None

    def forward(self, inputs, matrix_mapping):
        # input is a dictionary containing the image, the smpl parameters, and other info 
        # it is of shape (batch_size, key_values) where key values is of different dimensions
        
        print("model forward")
        image = inputs['masked_image'].cuda()
        print("image shape:", image.shape)

        gender = inputs['metadata']['gender']
        print("gender", gender)
        betas = inputs['betas']
        print("betas shape:", betas.shape)
        scale = inputs['smpl_params'][:, 0].cuda()
        smpl_pose = inputs["pose"].cuda()
        intrinsics = inputs['intrinsics'].cuda()
        smpl_tfs = inputs['smpl_tfs'].cuda()
        print("smpl_tfs shape:", smpl_tfs.shape)

        # The active voxels are mapped by the matrix_mapping (while in the canonical space)
        # TODO: Implement the mapping of the active voxels by the matrix_mapping 
        active_voxels_coo = self.voxel_mapping(matrix_mapping)   # Returns the active voxels coordinates in the canonical space
        print("active_voxels_coo shape:", active_voxels_coo.shape)

        # The active voxels are transformed to the dynamical space from the canonical space using the smpl pose parameters
        # TODO: Implement the transformation of the active voxels to the dynamical space using the smpl pose parameters
        # dynamical_voxels_coo = transform_voxels(active_voxels_coo, smpl_tfs)   # Returns the active voxels coordinates in the dynamical space

        dynamical_voxels_coo = self.dynamical_tfs_from_canonical(active_voxels_coo, smpl_tfs, betas, gender)   # To be tested if it works
        print("dynamical_voxels_coo shape:", dynamical_voxels_coo.shape)

        # ---
        # The feature encoder produces a pixel/voxel-wise feature embedding used to compute the occupancy field of the active voxels
        features = self.feature_encoder(image)   # (batch_size, features, img_width, img_height)
        print("features shape:", features.shape)

        # The feature embedding is interpolated to the active voxels in the dynamical space
        # TODO: Implement the interpolation of the feature embedding to the active voxels in the dynamical space (coordinates to pixel/voxel aligned features)
        features_at_voxels = self.interpolate_features(features, dynamical_voxels_coo)
        print("features_at_voxels shape:", features_at_voxels.shape)

        # The occupancy fields infers all the active voxels in the dynamical space  using the feature embedding
        # TODO: implement to run faster       
        occupancy_field = self.classifier(features_at_voxels)
        print("occupancy_field shape:", occupancy_field.shape)

        # The output of the occupancy field is used to compute the volume if needed
        # TODO: Implement the computation of the volume using the occupancy field as a separate method of the model, which can be called from outside

        # Also, the output of the occupancy field is used to refine the active voxels by modifying the matrix_mapping
        # The matrix mapping is to be kept larger than the actually occupied voxels, to allow for garment details etc.
        # However, where the active voxels are repeatedly not occupied, the matrix_mapping is to be reduced
        # The resolution (number of active voxels) reduces, while the number of occupied voxels remains the same (propotionally)
        # When the resolution is reduced enough, the matrix_mapping is to be refined to smaller voxels
        # TODO: implement the refinement of the matrix_mapping using the occupancy field as a separate method of the model, which can be called from outside
        # TODO: alternatively, make the matrix_mapping a property of the model, and check if refinement is due in the forward method

        return occupancy_field

    def interpolate_features(self, features, matrix_mapping):
        
        return interpolated_features

    def flatten_features(self, interpolated_features, matrix_mapping):

        return flattened_features_selected
    
    def voxel_mapping(self, matrix_mapping):
        active_voxels_coo = torch.nonzero(matrix_mapping == 1).float()
        print("active_voxels_coo shape:", active_voxels_coo.shape)

        for i in range(active_voxels_coo.shape[-1]):
            dim_min = 0
            dim_max = matrix_mapping.shape[i] - 1
            active_voxels_coo[:, i] = 2 * (active_voxels_coo[:, i] - dim_min) / (dim_max - dim_min) - 1
        
        print("active_voxels_coo shape:", active_voxels_coo.shape)
        print("active_voxels_coo:", active_voxels_coo)

        return active_voxels_coo
    
    def dynamical_tfs_from_canonical(self, points_coo, smpl_tfs, betas, gender):
        # points_coo = torch.tensor(points_coo).cuda().float()   # Suppose points_coo is a tensor of shape (batch_size, num_points, 3)
        weights = torch.zeros(len(gender))
        for i, gen in enumerate(gender):   # Each frame in the batch needs to have the skinning weights loaded separately
            print("gender i:", type(gen))
            print("betas i:", betas[i].shape)
            deformer = SMPLDeformer(betas=betas[i], gender=gen)
            print('check')
            weights[i] = deformer.query_weights(points_coo)
            print("weights i shape:", deformer.query_weights(points_coo).shape)
        print("weights shape:", weights.shape)
        verts_deformed = skinning(points_coo.unsqueeze(0), weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed