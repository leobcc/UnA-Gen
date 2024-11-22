import os
import numpy as np

def check_smpl_data(video_folder_path):
    # Check the data in the SMPL folder
    print('Checking SMPL data...')

    smplx_path = os.path.join(video_folder_path, 'smplx.npy')
    smplx = np.load(smplx_path, allow_pickle=True)
    print("SMPLX dictionary: ", smplx.item().keys())

    for key in smplx.item().keys():
        print(key, smplx.item()[key].shape)
        print(smplx.item()[key])


if __name__ == '__main__':
    # Check if the data is correct
    print('Checking data...')
    
    video_folder = '/home/lbocchi/UnA-Gen/data/data/train/0012_09'
    # Check the SMPL data
    check_smpl_data(video_folder)