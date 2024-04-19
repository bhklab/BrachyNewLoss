# Near-to-Target-aware OARs Segmentation in Cervical HDR Brachytherapy via Deep Learning
Using nnU-Net and distance-penalized loss functions to auto-segment OARs in cervical cancer brachytherapy


## nnU-Net preprocess
1. Create ‘Taskxx_gyn’ folder under ‘nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data’

2. Create 'Taskxx_gyn/imagesTs' folder and put the files you want to infer in the folder

3. Rename files as ended with ‘_0000’ using 
```
nnUNet_convert_decathlon_task -i [path of ‘Taskxx_gyn’]
```

## Create distance map 
1. Step 1: Run create-distance-map\calculate_distance_map_newnorm_step1_USE.py

2. Step 2: Run create-distance-map\calculate_distance_map_weighted_step2_USE.py

3. Move the files in Step 2 into '/imageTs' folder

4. Remove HR-CTV from labels: Run create-distance-map\remove_label_class.py

## Inference

```
nnUNet_predict -i [imagesTs] -o [inference folder] -t [task number] -m 3d_fullres -tr nnUNetTrainerV2_OAR_distDAv2mirror_noDS_DPCE -f all -p nnUNetPlansv2.1_ch1
```