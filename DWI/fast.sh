#!/bin/bash
for f in sub-9002_ses-2_dwi.nii.gz sub-9003_ses-2_dwi.nii.gz sub-9004_ses-2_dwi.nii.gz sub-9005_ses-2_dwi.nii.gz sub-9008_ses-2_dwi.nii.gz sub-9009_ses-2_dwi.nii.gz sub-9011_ses-2_dwi.nii.gz sub-9014_ses-2_dwi.nii.gz sub-9018_ses-2_dwi.nii.gz sub-9020_ses-2_dwi.nii.gz sub-9023_ses-2_dwi.nii.gz sub-9025_ses-2_dwi.nii.gz sub-9026_ses-2_dwi.nii.gz sub-9028_ses-2_dwi.nii.gz sub-9029_ses-2_dwi.nii.gz sub-9032_ses-2_dwi.nii.gz sub-9033_ses-2_dwi.nii.gz sub-9034_ses-2_dwi.nii.gz sub-9036_ses-2_dwi.nii.gz sub-9038_ses-2_dwi.nii.gz sub-9039_ses-2_dwi.nii.gz sub-9040_ses-2_dwi.nii.gz sub-9041_ses-2_dwi.nii.gz sub-9042_ses-2_dwi.nii.gz sub-9045_ses-2_dwi.nii.gz sub-9046_ses-2_dwi.nii.gz sub-9047_ses-2_dwi.nii.gz sub-9048_ses-2_dwi.nii.gz sub-9049_ses-2_dwi.nii.gz sub-9055_ses-2_dwi.nii.gz sub-9058_ses-2_dwi.nii.gz sub-9061_ses-2_dwi.nii.gz sub-9062_ses-2_dwi.nii.gz sub-9064_ses-2_dwi.nii.gz sub-9065_ses-2_dwi.nii.gz sub-9068_ses-2_dwi.nii.gz sub-9069_ses-2_dwi.nii.gz sub-9071_ses-2_dwi.nii.gz sub-9072_ses-2_dwi.nii.gz sub-9075_ses-2_dwi.nii.gz sub-9079_ses-2_dwi.nii.gz sub-9080_ses-2_dwi.nii.gz sub-9081_ses-2_dwi.nii.gz sub-9084_ses-2_dwi.nii.gz sub-9085_ses-2_dwi.nii.gz sub-9086_ses-2_dwi.nii.gz sub-9087_ses-2_dwi.nii.gz sub-9088_ses-2_dwi.nii.gz sub-9089_ses-2_dwi.nii.gz sub-9092_ses-2_dwi.nii.gz sub-9093_ses-2_dwi.nii.gz sub-9094_ses-2_dwi.nii.gz sub-9096_ses-2_dwi.nii.gz sub-9098_ses-2_dwi.nii.gz sub-9100_ses-2_dwi.nii.gz;
do

fast -b -B $f
rm *mixeltype.nii.gz
rm *pveseg.nii.gz
rm *pve*
rm *seg.nii.gz

done 