import numpy as np
# from medpy import metric
import nibabel as nib
from statistics import mean
from scipy import ndimage
import numpy as np
import nibabel as nib
import os
from scipy import ndimage

base_dir = '/home/ronin/gjbWork/GynSegRink/nnUNetData/nnUNet_raw_data/Task055_gyn'
label_path =  os.path.join(base_dir, 'labelsTs_cl5') # label only has the organ we want to segment
dist_step1_path = os.path.join(base_dir, 'distsTs_step1')
save_path = os.path.join(base_dir,'distsTs_step2')
"""
label_path = '/cluster/projects/radiomics/Gyn_Autosegmentation/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task057_gyn/labelsTs_cl4' # label only has the organ we want to segment
dist_step1_path = '/cluster/projects/radiomics/Gyn_Autosegmentation/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task057_gyn/distsTs_step1'
save_path = '/cluster/projects/radiomics/Gyn_Autosegmentation/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task057_gyn/distsTs_step2'
"""

power = 3
# def add_weights_to_dist_old(old_dist):
#     add_one = old_dist + 1 # old_dist: [0,1], add_one: [1,2]
#     nonone_min = np.min(add_one[add_one>1])
#     nonone_max = np.max(add_one)
#     expand_map = np.where(add_one>= nonone_min + 0.3*(nonone_max-nonone_min), add_one**2, add_one)
#     expand_map = np.where(expand_map>=(nonone_min + 0.5*(nonone_max-nonone_min))**2, np.sqrt(expand_map)**3, expand_map) # add more weights to >0.5 in old_dist
#     return expand_map

def add_weights_to_dist(old_dist, power):
    add_one = old_dist + 1 # old_dist: [0,1], add_one: [1,2]
    expand_map = add_one**power
    
    return expand_map

isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)
    print("The new save path is created!")

for f in os.listdir(label_path):
    labelname = os.path.join(label_path, f)
    label_ori = nib.load(labelname)
    label_cl4 = label_ori.get_fdata() 
    f1 = f[:-7] + '_0001.nii.gz'
    distname = os.path.join(dist_step1_path, f1)
    dist_cl4 = nib.load(distname)
    dist_cl4 = dist_cl4.get_fdata() # this dist map is for all OARs

    # for_use_dist = old_dist*label # only select the organ we want to segment
    # weighted_new_dist = add_weights_to_dist(for_use_dist)
    for m in range(1,len(np.unique(label_cl4))):
        labeltemp = np.copy(label_cl4)
        label1 = np.where(labeltemp!=m, 0, labeltemp)
        label1 = np.where(labeltemp==m, 1, label1)
        dist_label1 = dist_cl4*label1
        weighted_dist_label1 = add_weights_to_dist(dist_label1, power=power)
        weighted_dist_label1 = np.where(weighted_dist_label1==1, 0, weighted_dist_label1)  
        
        if m==1:
            new_weight_dist = weighted_dist_label1.copy()
        else:
            new_weight_dist = new_weight_dist+weighted_dist_label1
    new_weight_dist = np.where(new_weight_dist==0, 1, new_weight_dist) 

    savename = save_path + '/' + f[:-7] + '_0001.nii.gz'
    save_dist_map = nib.Nifti1Image(new_weight_dist, label_ori.affine)
    nib.save(save_dist_map, savename)
    print('Saved', f)
