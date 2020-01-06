#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###default parameter settings

###task-fmri info
TR = 0.72
atlas_name = 'MMP'
##task_modality = 'motor'
'''
pathdata = "/home/yu/PycharmProjects/HCP_data/aws_s3_HCP1200/"
pathout = "/home/yu/PycharmProjects/HCP_data/temp_res_new2/"
pathatlas = pathdata + "../codes/HCP_S1200_GroupAvg_v1/"
'''
pathdata = "/data/cisl/yuzhang/projects/HCP/aws_s3_HCP1200/"
pathout = "/data/cisl/yuzhang/projects/HCP/temp_res_new2/"
pathatlas = pathdata + "../codes/HCP_S1200_GroupAvg_v1/"


##data pipe
#block_dura = 17
TR_step = 1
nr_thread = 2
buffersize = 4
batch_size = 128

##model setting
test_sub_num = 0
test_size = 0.2
val_size = 0.1
randseed = 1234

USE_GPU_CPU = 1
num_CPU = 6
num_GPU = 1

###brain graph setting
adj_mat_type = 'RSFC'
coarsening_levels = 1

###
if atlas_name == 'MMP':
    atlas_name = 'MMP'
    mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
elif atlas_name == 'BNA':
    atlas_name = 'BNA'
    mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "BN_Atlas_246_2mm.nii.gz"
elif atlas_name == 'CBIG1k':
    atlas_name = 'CBIG1k'
    mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"
elif atlas_name == 'CBIG400':
    atlas_name = 'CBIG400'
    mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "Schaefer2018_400Parcels_17Networks_order.dlabel.nii"

if adj_mat_type == 'surf':
    adj_mat_file = pathout + atlas_name + '_adjacency_mat_white.pconn.nii'
elif adj_mat_type == 'SC':
    adj_mat_file = pathatlas + 'S1200.All.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'
elif adj_mat_type == 'RSFC':
    adj_mat_file = pathout + '_'.join(('rsfmri', 'REST1', 'LR', atlas_name, "ROI_act_1200R_test2_March2019_new.lmdb"))
