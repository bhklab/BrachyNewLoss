import numpy as np
import nibabel as nib
import os
from scipy import ndimage


base_dir = '/home/ronin/gjbWork/GynSegRink/nnUNetData/nnUNet_raw_data/Task055_gyn'
path = os.path.join(base_dir, 'labelsTs_cl5')
save_path = os.path.join(base_dir,'labelsTs_cl4')


isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)
    print("The new save path is created!")


for f in os.listdir(path):
    filename = path + '/' + f
    label_ori_nib = nib.load(filename)
    label_ori = label_ori_nib.get_fdata()
    label_save = label_ori.copy()
    
    '''To save 4 OARs only - remove HR-CTV'''
    label_save = np.where(label_save==5,0,label_save)

    '''To save HR-CTV only'''
    # label_save = np.where(label_save!=5,0,label_save)
    # label_save = np.where(label_save==5,1,label_save)

    '''save as .nii.gz file'''
    label_save = nib.Nifti1Image(label_save, label_ori_nib.affine)
    savename = save_path + '/' + f
    nib.save(label_save, savename)
    print(f + ' is calculated!')
