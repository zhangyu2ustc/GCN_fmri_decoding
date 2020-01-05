#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###default parameter settings

###task-fmri info
TR = 0.72
atlas_name = 'MMP'
task_modality = 'MOTOR'
pathdata = "/home/yu/PycharmProjects/HCP_data/aws_s3_HCP1200/"
pathout = "/home/yu/PycharmProjects/HCP_data/temp_res_new2/"
mmp_atlas = pathdata + "codes/HCP_S1200_GroupAvg_v1/"+"Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"

##data pipe
block_dura = 17
TR_step = 1
nr_thread = 2
buffersize = 4

##model setting
test_sub_num = 0
test_size = 0.2
val_size = 0.1
randseed = 1234

USE_GPU_CPU = 1
num_CPU = 6
num_GPU = 1

###brain graph setting
adj_mat_file = pathdata + 'codes/MMP_adjacency_mat_white.pconn.nii'
adj_mat_type = 'surf'
coarsening_levels = 1