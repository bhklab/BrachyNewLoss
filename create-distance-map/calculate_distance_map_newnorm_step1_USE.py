import numpy as np
import nibabel as nib
import os
from scipy import ndimage
import matplotlib.pyplot as plt

# label folder

path = '/home/ronin/gjbWork/GynSegRink/nnUNetData/nnUNet_raw_data/Task055_gyn/labelsTs_cl5'
# save folder
save_path = '/home/ronin/gjbWork/GynSegRink/nnUNetData/nnUNet_raw_data/Task055_gyn/distsTs_step1'

def NormalizeData(data):
    norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm

isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)
    print("The new save path is created!")

def compute_edts_forPenalizedLoss(label_raw):
    '''calculate distance penalty map (I) based on HR-CTV'''
    # isolate HR-CTV 
    label_raw[label_raw<5] = 0
    label_raw[label_raw == 5] = 1
    # calculae DTM of 3D label (real label)
    label_re = 1-label_raw
    label_dist = ndimage.distance_transform_edt(label_re)
    # normalize label to [1,2]
    max_dist = np.amax(label_dist)
    label_dist_norm = label_dist/max_dist + 1
    # calculate penalty map using inverse square law (I = 1/R^2)
    I = 1/(label_dist_norm**2)
    # normalize I to [0,1]
    I = NormalizeData(I)
    I = np.where(I==1, 0, I)
    return I
        
for f in os.listdir(path):
    labelname = os.path.join(path, f)
    label_ori = nib.load(labelname)
    label = label_ori.get_fdata()
    label_copy = label.copy()
    dist = compute_edts_forPenalizedLoss(label_copy) 

    label_use = np.where(label==5, 0, label)

    '''normalize the ratio based on each organ'''
    # new_dist_map = np.zeros(label.shape)
    for m in range(1,len(np.unique(label_use))):
        labeltemp = np.copy(label_use)
        label1 = np.where(labeltemp!=m, 0, labeltemp)
        label1 = np.where(labeltemp==m, 1, label1)
        label_dist1 = label1*dist
        label_dist1 = NormalizeData(label_dist1)
        
        if m==1:
            new_dist_map = label_dist1.copy()
        else:
            new_dist_map = new_dist_map+label_dist1

    savename = save_path + '/' + f[:-7] + '_0001.nii.gz'
    save_dist_map = nib.Nifti1Image(new_dist_map, label_ori.affine)
    nib.save(save_dist_map, savename)
    print('Saved', f)