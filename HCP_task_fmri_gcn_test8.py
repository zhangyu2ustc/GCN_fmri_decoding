#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

## ps | grep python; pkill python

from pathlib import Path
import glob
import itertools
import lmdb
import h5py
import os
import sys
import time
import datetime
import shutil
from operator import itemgetter
from collections import Counter

import ast
import math
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from nilearn import connectome
from nilearn import signal,image,masking
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from keras.utils import np_utils

import pickle
import lmdb
import tensorflow as tf
from tensorpack import dataflow
from tensorpack.utils.serialize import dumps, loads
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.common import get_tf_version_tuple

try:
    # import cnn_graph
    ###sys.path.append('/path/to/application/app/folder')
    from cnn_graph.lib_new import models, graph, coarsening, utils
except ImportError:
    print('Could not find the package of graph-cnn ...')
    print('Please check the location where cnn_graph is !\n')

#####global variable settings
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''
USE_GPU_CPU = 1
num_CPU = 6
num_GPU = 1 #get_num_gpu()  # len(used_GPU_avail)
print('\nAvaliable GPUs for usage: %d \n' % num_GPU)
config_TF = tf.ConfigProto(intra_op_parallelism_threads=num_CPU,\
        inter_op_parallelism_threads=num_CPU, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config_TF)
tf.keras.backend.set_session(session)

global TR, dataarg, droprate
TR = 0.72
dataarg = 1
##AtlasName = 'MMP'
#adj_mat_type = 'SC' #'surface'

'''
pathdata = '/data/cisl/yuzhang/projects/HCP/'
###pathdata = '/home/yuzhang/scratch/HCP/'
pathout = pathdata + "temp_res_new2/"
#mmp_atlas = pathdata + "codes/HCP_S1200_GroupAvg_v1/"+"Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
#lmdb_filename = pathout+modality+"_MMP_ROI_act_1200R_test_Dec2018_ALL.lmdb"
#adj_mat_file = pathdata + 'codes/MMP_adjacency_mat_white.pconn.nii'

pathdata="/home/yuzhang/scratch/HCP/" ##aws_s3_HCP1200/FMRI/tfMRI_MOTOR_LR/
pathatlas=pathdata + "codes/HCP_S1200_GroupAvg_v1/"
pathout=pathdata+"temp_res_new2/"
#mmp_atlas = pathdata + "HCP_S1200_GroupAvg_v1/"+"Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
##lmdb_filename = pathout+modality+"_MMP_ROI_act_1200R_test_Dec2018_ALL.lmdb"
##adj_mat_file = pathout + 'MMP_adjacency_mat_white.pconn.nii'
'''
pathdata="/home/yu/PycharmProjects/HCP_data/" ##aws_s3_HCP1200/FMRI/tfMRI_MOTOR_LR/
pathatlas=pathdata + "HCP_S1200_GroupAvg_v1/"
pathout=pathdata+"temp_res_new2/"

##############################################
###################################################
def bulid_dict_task_modularity(modality):
    ###edited Jan 25th 2019: we need to use block design for fmri decoding when splitting trials into data samples
    ##build the dict for different subtypes of events under different modalities
    motor_task_con = {"rf": "footR_mot",
                      "lf": "footL_mot",
                      "rh": "handR_mot",
                      "lh": "handL_mot",
                      "t": "tongue_mot"}

    lang_task_ev =   {"present_math":  "pmath_lang",
                      #"question_math": "pmath_lang",
                      "response_math": "rmath_lang",
                      "present_story":  "pstory_lang",
                      #"question_story": "pstory_lang" ,
                      "response_story": "rstory_lang"}

    lang_task_con =  {"math":  "math_lang",
                      "story": "story_lang"}
    emotion_task_con={"fear": "fear_emo",
                      "neut": "non_emo"}

    gambl_task_ev =  {"win_event":  "win_gamb",
                      "loss_event": "loss_gamb",
                      "neut_event": "non_gamb"}

    gambl_task_con = {"win":  "win_gamb",
                      "loss": "loss_gamb"}

    reson_task_con = {"match":    "match_reson",
                      "relation": "relat_reson"}
    social_task_con ={"mental": "mental_soc",
                      "rnd":  "random_soc"}
    wm_task_con   =  {"2bk_body":   "body2b_wm",
                      "2bk_faces":  "face2b_wm",
                      "2bk_places": "place2b_wm",
                      "2bk_tools":  "tool2b_wm",
                      "0bk_body":   "body0b_wm",
                      "0bk_faces":  "face0b_wm",
                      "0bk_places": "place0b_wm",
                      "0bk_tools":  "tool0b_wm"}

    dicts = [motor_task_con, lang_task_con, emotion_task_con, reson_task_con, social_task_con, wm_task_con] ##gambl_task_con,
    from collections import defaultdict
    all_task_con = defaultdict(list)  # uses set to avoid duplicates
    for d in dicts:
        for k, v in d.items():
            #all_task_con[k].append(v)  ## all_task_con[k]=v to remove the list []
            all_task_con[k] = v

    rwm_task_con = defaultdict(list)
    for d in [wm_task_con, reson_task_con]:
        for k, v in d.items():
            rwm_task_con[k] = v

    mod_chosen = modality[:3].lower().strip()
    mod_choices = {'mot': 'MOTOR',
                   'lan': 'LANGUAGE',
                   'ela': 'LANGUAGE',
                   'emo': 'EMOTION',
                   'ega': 'GAMBLING',
                   'gam': 'GAMBLING',
                   'rel': 'RELATIONAL',
                   'soc': 'SOCIAL',
                   'wm':  'WM',
                   'rwm': 'RWM',
                   'all': 'ALLTasks'}
    task_choices = {'mot': motor_task_con,
                    'lan': lang_task_con,
                    'ela': lang_task_ev,
                    'emo': emotion_task_con,
                    'gam': gambl_task_con,
                    'ega': gambl_task_ev,
                    'rel': reson_task_con,
                    'soc': social_task_con,
                    'wm': wm_task_con,
                    'rwm': rwm_task_con,
                    'all': all_task_con}

    modality = mod_choices.get(mod_chosen, 'default')
    task_contrasts = task_choices.get(mod_chosen, 'default')
    return task_contrasts, modality

#####start collecting data for classification algorithm
def load_fmri_data(pathdata,modality=None,confound_name=None):
    ###fMRI decoding: using event signals instead of activation pattern from glm
    ##collect task-fMRI signals

    if not modality:
        modality = 'MOTOR'  # 'MOTOR'

    pathfmri = pathdata + 'aws_s3_HCP1200/FMRI/'
    print("Loading fmri data from data folder:",pathfmri)
    pathdata = Path(pathfmri)
    subjects = []
    fmri_files = []
    confound_files = []
    if modality == 'ALLTasks' or modality == 'RWM':
        for fmri_file in sorted(pathdata.glob('tfMRI_*_??/*tfMRI_*_??_Atlas.dtseries.nii')):
            subjects.append(Path(os.path.dirname(fmri_file)).parts[-3])
            fmri_files.append(str(fmri_file))

        for confound in sorted(pathdata.glob('tfMRI_*_??/*Movement_Regressors.txt')):
            confound_files.append(str(confound))
    else:
        for fmri_file in sorted(pathdata.glob('tfMRI_'+modality+'_??/*tfMRI_'+modality+'_??_Atlas.dtseries.nii')):
            subjects.append(Path(os.path.dirname(fmri_file)).parts[-3])
            fmri_files.append(str(fmri_file))

        for confound in sorted(pathdata.glob('tfMRI_'+modality+'_??/*Movement_Regressors.txt')):
            confound_files.append(str(confound))

    print('Included {} fmri data and {} confound files in the dataset!\n'.format(len(fmri_files),len(confound_files)))
    return fmri_files, confound_files, subjects


def load_event_files(pathdata, modality, fmri_files, confound_files, ev_filename=None, flag_event=False):
    ###collect the event design files
    confound = np.loadtxt(confound_files[0])
    Subject_Num = len(confound_files)
    Trial_Num = confound.shape[0]

    pathfmri = pathdata + 'aws_s3_HCP1200/FMRI/'
    pathdata = Path(pathfmri)
    if modality == 'ALLTasks' or modality == 'RWM':
        modality_str = '*'
    else:
        modality_str = modality
    if flag_event:
        modality_str2 = modality_str + '_event'
    else:
        modality_str2 = modality_str

    EVS_files = []
    subj = 0
    '''
    for ev, sub_count in zip(sorted(pathdata.glob('tfMRI_' + modality + '_??/*combined_events_spm_' + modality + '.csv')),range(Subject_Num)):
        ###remove fmri files if the event design is missing
        while os.path.dirname(fmri_files[subj]) < os.path.dirname(str(ev)):
            print("Event files and fmri data are miss-matching for subject: ")
            print(Path(os.path.dirname(str(ev))).parts[-3::2], ':',
                  Path(os.path.dirname(fmri_files[subj])).parts[-3::2])
            print("Due to missing event files for subject : %s" % os.path.dirname(fmri_files[subj]))
            fmri_files[subj] = []
            confound_files[subj] = []
            subj += 1
            if subj > Subject_Num:
                break
        if os.path.dirname(fmri_files[subj]) == os.path.dirname(str(ev)):
            EVS_files.append(str(ev))
            subj += 1
    '''
    ###adjust the code after changing to the new folder
    for ev in sorted(pathdata.glob('tfMRI_' + modality_str + '_??/*combined_events_spm_' + modality_str2 + '.csv')):
        ###remove fmri files if the event design is missing

        ##not including the event design, use block design instead
        if not flag_event and 'event' in os.path.basename(str(ev)).split('_')[-1]:  continue;

        while os.path.basename(confound_files[subj]).split('_')[0] < os.path.basename(str(ev)).split('_')[0]:
            print("Event files and fmri data are miss-matching for subject: ")
            print(os.path.basename(str(ev)).split('_')[0], ':', os.path.basename(confound_files[subj]).split('_')[0])
            print("Due to missing event files for subject : %s_%s" % (Path(os.path.dirname(confound_files[subj])).parts[-1], os.path.basename(confound_files[subj]).split('_')[0]))
            ##fmri_files[subj] = []
            confound_files[subj] = []
            subj += 1
            if subj > Subject_Num:
                break
        if os.path.basename(confound_files[subj]).split('_')[0] == os.path.basename(str(ev)).split('_')[0]:
            EVS_files.append(str(ev))
            subj += 1

    fmri_files = list(filter(None, fmri_files))
    confound_files = list(filter(None, confound_files))
    if len(EVS_files) != len(confound_files):
        print('Miss-matching number of subjects between event:{} and fmri:{} files'.format(len(EVS_files), len(confound_files)))

    print("Data samples including {} subjects with {} trials for event design. \n".format(len(EVS_files),Trial_Num))
    ################################
    ###loading all event designs
    if not ev_filename:
        ev_filename = "_event_labels_1200R_LR_RL_new2.txt"
    if flag_event:
        print("Using event design files for task:  ", modality)
        ev_filename = ev_filename.replace('.h5', '_event.h5')

    events_all_subjects_file = pathout+modality+ev_filename
    if os.path.isfile(events_all_subjects_file):
        trial_infos = pd.read_csv(EVS_files[0],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
        Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)

        print('Collecting trial info from file:', events_all_subjects_file)
        subjects_trial_labels = pd.read_csv(events_all_subjects_file,sep="\t",encoding="utf8")
        print(subjects_trial_labels.keys())

        try:
            label_matrix = subjects_trial_labels['label_data'].values
            # print(label_matrix[0],label_matrix[1])
            # xx = label_matrix[0].split(",")
            subjects_trial_label_matrix = []
            for subi in range(len(label_matrix)):
                xx = [x.replace("['", "").replace("']", "") for x in label_matrix[subi].split("', '")]
                subjects_trial_label_matrix.append(xx)
            subjects_trial_label_matrix = pd.DataFrame(data=(subjects_trial_label_matrix))
        except:
            print('only extracting {} trials from event design'.format(Trial_Num))
            subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]

        #subjects_trial_label_matrix = subjects_trial_labels.values.tolist()
        trialID = subjects_trial_labels['trialID']
        sub_name = subjects_trial_labels['subject'].tolist()
        coding_direct = subjects_trial_labels['coding']
        print(np.array(subjects_trial_label_matrix).shape,len(sub_name),len(np.unique(sub_name)),len(coding_direct))
    else:
        print('Loading trial info for each task-fmri file and save to csv file:', events_all_subjects_file)
        subjects_trial_label_matrix = []
        sub_name = []
        coding_direct = []
        modality_pre = ''
        for subj in np.arange(len(EVS_files)):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            #Trial_Num = nib.load(fmri_files[subj]).shape[-1]
            Trial_Num = np.loadtxt(confound_files[subj]).shape[0]
            if os.path.basename(pathsub) != modality_pre:
                print("Start analysizing event data for modality {} ".format(os.path.basename(pathsub)))
                modality_pre = os.path.basename(pathsub)

            #sub_name.append(pathsub.parts[-3])
            ###adjust the code after changing to the new folder
            sub_name.append(str(os.path.basename(EVS_files[subj]).split('_')[0]))
            coding_direct.append(pathsub.parts[-1].replace('tfMRI_',''))

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
            Onsets = np.array((trial_infos.onset/TR).astype(int)-1) #(trial_infos.onset/TR).astype(int)
            Duras = np.array((trial_infos.duration/TR).astype(int)) #(trial_infos.duration/TR).astype(int)
            Movetypes = list(trial_infos.task)
            move_mask = pd.Series(Movetypes).isin(task_contrasts.keys())
            Onsets = Onsets[move_mask]
            Duras = Duras[move_mask]
            Movetypes = [Movetypes[i] for i in range(len(move_mask)) if move_mask[i]]
            event_len = Onsets[-1] + Duras[-1] + 1   ###start with 0
            while event_len > Trial_Num:
                '''
                ##del Onsets[-1];  del Duras[-1];  del Movetypes[-1]
                print('Cutting {} event design due to incomplete scanning...short of {} volumes out of {}'.format(os.path.basename(pathsub),event_len-Trial_Num,Trial_Num))
                try:
                    Duras[-1] -= (event_len-Trial_Num)
                except:
                    print(Duras,event_len,Trial_Num)
                event_len -= (event_len-Trial_Num)
                if Duras[-1] <= 0:
                '''
                #print('Remove last trials due to incomplete scanning...short of {} volumes out of {}'.format(event_len - Trial_Num, Trial_Num))
                Onsets=Onsets[:-1];  Duras=Duras[:-1];  del Movetypes[-1]
                event_len = Onsets[-1] + Duras[-1] + 1

            labels = ["rest"]*Trial_Num;
            trialID = [0] * Trial_Num;
            tid = 1
            for start,dur,move in zip(Onsets,Duras,Movetypes):
                for ti in range(start-1,start+dur):
                    try:
                        labels[ti]= task_contrasts[move]
                        trialID[ti] = tid
                    except:
                        print('Error loading for event file {}'.format(EVS_files[subj]))
                        print('Run in to #{} volume when having total {} volumes'.format(ti,Trial_Num))
                        print(Onsets, Duras, Movetypes, move)
                tid += 1
            subjects_trial_label_matrix.append(labels)

        ##subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)
        print(np.array(subjects_trial_label_matrix).shape)
        sub_name = list(map("_".join, zip(sub_name,coding_direct)))
        #print(np.array(subjects_trial_label_matrix[0]))
        try:
            subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),columns=['label_data'])
        except:
            subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix), columns=['trial' + str(i + 1) for i in range(Trial_Num)])
        subjects_trial_labels['trialID'] = tid-1
        subjects_trial_labels['subject'] = sub_name
        subjects_trial_labels['coding'] = coding_direct
        subjects_trial_labels.keys()
        #print(subjects_trial_labels['subject'],subjects_trial_labels['coding'])

        ##save the labels
        subjects_trial_labels.to_csv(events_all_subjects_file,sep='\t', encoding='utf-8',index=False)

    block_dura = np.unique(Duras)[0]
    return subjects_trial_label_matrix, sub_name, block_dura

def load_fmri_data_from_lmdb(lmdb_filename):
    ##lmdb_filename = pathout + modality + "_MMP_ROI_act_1200R_test_Dec2018_ALL.lmdb"
    ## read lmdb matrix
    print('loading data from file: %s' % lmdb_filename)
    matrix_dict = []
    fmri_sub_name = []
    if not os.path.isfile(lmdb_filename) and modality == 'ALLTasks':
        print("Loading fMRI data from all tasks and merge into one lmdb file:", lmdb_filename)
        lmdb_env = lmdb.open(lmdb_filename, subdir=False,readonly=False, map_size=int(1e12) * 2,  meminit=False, map_async=True)
        write_frequency = 100

        pathout = Path(os.path.dirname(lmdb_filename))
        for lmdb_mod in sorted(pathout.glob(os.path.basename(lmdb_filename).replace(modality,'*'))):
            mod_name = os.path.basename(lmdb_mod).split('_')[0]
            if mod_name == 'ALLTasks': continue
            print('Loading data for modality',mod_name)
            lmdb_txn = lmdb_env.begin(write=True)

            lmdb_mod_env = lmdb.open(str(lmdb_mod), subdir=False, readonly=True)
            with lmdb_mod_env.begin() as lmdb_mod_txn:
                mod_cursor = lmdb_mod_txn.cursor()
                for idx,(key, value) in enumerate(mod_cursor):
                    lmdb_txn.put(key, value)
                    if (idx + 1) % write_frequency == 0:
                        lmdb_txn.commit()
                        lmdb_txn = lmdb_env.begin(write=True)

            lmdb_txn.commit()
            lmdb_mod_env.close()

            lmdb_env.sync()
        lmdb_env.close()

    ##########################################33
    lmdb_env = lmdb.open(lmdb_filename, subdir=False)
    try:
        lmdb_txn = lmdb_env.begin()
        listed_fmri_files = loads(lmdb_txn.get(b'__keys__'))
        listed_fmri_files = [l.decode("utf-8") for l in listed_fmri_files]
        print('Stored fmri data from files:')
        print(len(listed_fmri_files))
    except:
        print('Search each key for every fmri file...')

    with lmdb_env.begin() as lmdb_txn:
        cursor = lmdb_txn.cursor()
        for key, value in cursor:
            # print(key)
            if key == b'__keys__':
                continue
            pathsub = Path(os.path.dirname(key.decode("utf-8")))
            ##subname_info = os.path.basename(key.decode("utf-8")).split('_')
            ##fmri_sub_name.append('_'.join((subname_info[0], subname_info[2], subname_info[3])))
            #############change due to directory switch to projects
            subname_info = str(Path(os.path.dirname(key.decode("utf-8"))).parts[-3])
            fmri_sub_name.append(Path(os.path.dirname(key.decode("utf-8"))).parts[-1].replace('tfMRI',subname_info))
            data = loads(lmdb_txn.get(key)).astype('float32', casting='same_kind')
            matrix_dict.append(np.array(data))
    lmdb_env.close()

    return matrix_dict, fmri_sub_name

def load_rsfmri_data_matrix(lmdb_filename,Trial_Num=1200):
    import lmdb
    from tensorpack.utils.serialize import dumps, loads
    ## read lmdb matrix
    print('loading data from file: %s' % lmdb_filename)
    matrix_dict = []
    fmri_sub_name = []
    lmdb_env = lmdb.open(lmdb_filename, subdir=False)
    try:
        lmdb_txn = lmdb_env.begin()
        listed_fmri_files = loads(lmdb_txn.get(b'__keys__'))
        listed_fmri_files = [l.decode("utf-8") for l in listed_fmri_files]
        print('Stored fmri data from files:')
        print(len(listed_fmri_files))
    except:
        print('Search each key for every fmri file...')

    with lmdb_env.begin() as lmdb_txn:
        cursor = lmdb_txn.cursor()
        for key, value in cursor:
            # print(key)
            if key == b'__keys__':
                continue
            pathsub = Path(os.path.dirname(key.decode("utf-8")))
            if any('REST' in string for string in lmdb_filename.split('_')):
                fmri_sub_name.append(pathsub.parts[-3] + '_' + pathsub.parts[-1].split('_')[-2][-1] + '_' + pathsub.parts[-1].split('_')[-1])
            else:
                fmri_sub_name.append(pathsub.parts[-3] + '_' + pathsub.parts[-1].split('_')[-1])
            data = loads(lmdb_txn.get(key))
            if any('REST' in string for string in lmdb_filename.split('_')):
                if data is None or data.shape[-1] != Trial_Num:
                    print('fmri data shape mis-matching between subjects...')
                    print('Check subject:  %s with only %d Trials \n' % (fmri_sub_name[-1], data.shape[0]))
                    del fmri_sub_name[-1]
                else:
                    #print(np.array(data).shape)
                    matrix_dict.append(np.array(data))
            else:
                print('wrong located')
                matrix_dict.append(np.array(data))
    lmdb_env.close()
    print(np.array(matrix_dict).shape)
    return matrix_dict, fmri_sub_name

def preclean_data_for_shape_match_new(subjects_tc_matrix,subjects_trial_label_matrix, fmri_sub_name, ev_sub_name):
    print("Pre-clean the fmri and event data to make sure the matching shapes between two arrays!")
    Subject_Num = np.array(subjects_tc_matrix).shape[0]
    Trial_Num, Region_Num = subjects_tc_matrix[0].shape

    #####################sort list of files
    print('Sort both fmri and event files into the same order!')
    fmrifile_index, fmri_sub_name_sorted = zip(*sorted(enumerate(fmri_sub_name), key=itemgetter(1)))
    subjects_tc_matrix_sorted = [subjects_tc_matrix[ind] for ind in fmrifile_index]
    ev_index, ev_sub_name_sorted = zip(*sorted(enumerate(ev_sub_name), key=itemgetter(1)))
    subjects_trial_label_matrix_sorted = [list(filter(None, subjects_trial_label_matrix.iloc[ind])) for ind in ev_index]
    fmri_sub_name_sorted = list(fmri_sub_name_sorted)
    ev_sub_name_sorted = list(ev_sub_name_sorted)
    ####check matching of filenames
    for ev, subcount in zip(ev_sub_name_sorted, range(len(ev_sub_name_sorted))):
        evfile_mask = pd.Series(ev).isin(fmri_sub_name_sorted)
        if not evfile_mask[0]:
            print("Remove event file: {} from the list!!".format(ev))
            del subjects_trial_label_matrix_sorted[subcount]
            ev_sub_name_sorted.remove(ev)
    for fmri_file, subcount in zip(fmri_sub_name_sorted, range(len(fmri_sub_name_sorted))):
        fmrifile_mask = pd.Series(fmri_file).isin(ev_sub_name_sorted)
        if not fmrifile_mask[0]:
            print("Remove fmri file: {} from the list!!".format(fmri_file))
            del subjects_tc_matrix_sorted[subcount]
            fmri_sub_name_sorted.remove(fmri_file)
    print('New shapes of fmri-data-matrix and trial-label-matrix after matching!')
    print(np.array(subjects_tc_matrix_sorted).shape, np.array(subjects_trial_label_matrix_sorted).shape)

    if len(fmri_sub_name_sorted) != len(ev_sub_name_sorted):
        print('Warning: Mis-matching subjects list between fmri-data-matrix and trial-label-matrix')
        print(np.array(subjects_tc_matrix_sorted).shape, np.array(subjects_trial_label_matrix_sorted).shape)
        ###########matching each evfile in fmri data
        subjects_tc_matrix_new = []
        fmri_sub_name_new = []
        for ev in ev_sub_name_sorted:
            ###remove fmri files if the event design is missing
            fmrifile_mask = pd.Series(fmri_sub_name_sorted).isin([ev])
            if np.sum(fmrifile_mask):
                subjects_tc_matrix_new.append(subjects_tc_matrix_sorted[np.where(fmrifile_mask)[0][0]])
                fmri_sub_name_new.append(fmri_sub_name_sorted[np.where(fmrifile_mask)[0][0]])
        fmri_sub_name = fmri_sub_name_new
        subjects_tc_matrix = np.array(subjects_tc_matrix_new)

        ###########matching each fmri file in event data
        subjects_trial_label_matrix_new = []
        ev_sub_name_new = []
        for fmri_file in fmri_sub_name_sorted:
            ###remove fmri files if the event design is missing
            evfile_mask = pd.Series(ev_sub_name_sorted).isin([fmri_file])
            if np.sum(evfile_mask):
                subjects_trial_label_matrix_new.append(subjects_trial_label_matrix_sorted[np.where(evfile_mask)[0][0]])
                ev_sub_name_new.append(ev_sub_name_sorted[np.where(evfile_mask)[0][0]])
        ev_sub_name = ev_sub_name_new
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix_new)
    else:
        ev_sub_name = ev_sub_name_sorted
        subjects_trial_label_matrix = subjects_trial_label_matrix_sorted
        fmri_sub_name = fmri_sub_name_sorted
        subjects_tc_matrix = np.array(subjects_tc_matrix_sorted)

    for subj in range(min(len(fmri_sub_name), len(ev_sub_name))):
        try:
            tsize, rsize = np.array(subjects_tc_matrix[subj]).shape
            tsize2 = len(list(filter(None, subjects_trial_label_matrix[subj])))
        except:
            print(subj == Subject_Num - 1)
            print('The end of SubjectList...\n')
        if tsize != tsize2:
            '''
            if tsize2 > Trial_Num:
                ##print('Cut event data for subject %s from %d to fit event label matrix' % (fmri_sub_name[subj],tsize2))
                subjects_trial_label_matrix[subj][tsize:] = []
            '''
            print('Remove fmri_subject: {} and event subject {} due to different trial num: {} / {}'.format(fmri_sub_name[subj],ev_sub_name[subj],tsize,tsize2))
            del fmri_sub_name[subj]
            del ev_sub_name[subj]
            del subjects_tc_matrix[subj]
            del subjects_trial_label_matrix[subj]

        if rsize != Region_Num:
            print('Remove fmri_subject: {} and event subject {} due to different region num: {}/{}'.format(fmri_sub_name[subj], ev_sub_name[subj],rsize,Region_Num))
            del fmri_sub_name[subj]
            del ev_sub_name[subj]
            del subjects_tc_matrix[subj]
            del subjects_trial_label_matrix[subj]

    print('Done matching data shapes:', np.array(subjects_tc_matrix).shape, np.array(subjects_trial_label_matrix).shape)
    return subjects_tc_matrix, subjects_trial_label_matrix, fmri_sub_name


def preclean_data_for_shape_match(subjects_tc_matrix,subjects_trial_label_matrix, fmri_sub_name, ev_sub_name):
    print("Pre-clean the fmri and event data to make sure the matching shapes between two arrays!")
    Subject_Num = np.array(subjects_tc_matrix).shape[0]
    Trial_Num, Region_Num = subjects_tc_matrix[0].shape

    if len(fmri_sub_name) != len(ev_sub_name):
        print('Warning: Mis-matching subjects list between fmri-data-matrix and trial-label-matrix')
        print(np.array(subjects_tc_matrix).shape, np.array(subjects_trial_label_matrix).shape)
        subj = 0
        if len(fmri_sub_name) > len(ev_sub_name):
            for ev, subcount in zip(ev_sub_name, range(Subject_Num)):
                ###remove fmri files if the event design is missing
                while fmri_sub_name[subj].split('_')[0] < str(ev).split('_')[0]:
                    print("Event files and fmri data are miss-matching for subject: ")
                    print(ev, ':', fmri_sub_name[subj])
                    print("Due to missing event files for subject : %s" % fmri_sub_name[subj])
                    del fmri_sub_name[subj]
                    del subjects_tc_matrix[subj]
                    subj += 1
                else:
                    if subj > Subject_Num:
                        ev_sub_name.remove(ev)
                        del subjects_trial_label_matrix[subcount]
                        subj = subcount
                    if fmri_sub_name[subj] == str(ev): subj += 1
            subjects_tc_matrix[subj:] = []
            fmri_sub_name[subj:] = []

        elif len(fmri_sub_name) < len(ev_sub_name):
            for fmri_file, subcount in zip(fmri_sub_name, range(len(ev_sub_name))):
                ###remove fmri files if the event design is missing
                while str(ev_sub_name[subj]).split('_')[0] < fmri_file.split('_')[0]:
                    print("Event files and fmri data are miss-matching for subject: ")
                    print(ev_sub_name[subj], ':', fmri_file)
                    print("Due to missing fmri data for subject : %s" % str(ev_sub_name[subj]))
                    del ev_sub_name[subj]
                    del subjects_trial_label_matrix[subj]
                    subj += 1
                else:
                    if subj > len(ev_sub_name):
                        fmri_sub_name.remove(fmri_file)
                        del subjects_tc_matrix[subcount]
                        subj = subcount
                    if str(ev_sub_name[subj]) == fmri_file: subj += 1
            subjects_trial_label_matrix[subj:] = []
            ev_sub_name[subj:] = []

    for subj in range(min(len(fmri_sub_name), len(ev_sub_name))):
        try:
            tsize, rsize = np.array(subjects_tc_matrix[subj]).shape
            tsize2 = len(subjects_trial_label_matrix[subj])
        except:
            print(subj == Subject_Num - 1)
            print('The end of SubjectList...\n')
        if tsize != tsize2:
            '''
            if tsize2 > Trial_Num:
                ##print('Cut event data for subject %s from %d to fit event label matrix' % (fmri_sub_name[subj],tsize2))
                subjects_trial_label_matrix[subj][tsize:] = []
            '''
            print('Remove fmri_subject: {} and event subject{} due to different trial num: {} / {}'.format(fmri_sub_name[subj],ev_sub_name[subj],tsize,tsize2))
            del subjects_tc_matrix[subj]
            del subjects_trial_label_matrix[subj]

        if rsize != Region_Num:
            print('Remove subject: %s due to different region num: %d in the fmri data' % (fmri_sub_name[subj], rsize))
            del subjects_tc_matrix[subj]
            del subjects_trial_label_matrix[subj]

    print('Done matching data shapes:', np.array(subjects_tc_matrix).shape, np.array(subjects_trial_label_matrix).shape)
    return subjects_tc_matrix, subjects_trial_label_matrix


#####################################
###standard scaler for nd array instead of 2d matrix
from sklearn.base import TransformerMixin
class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = preprocessing.StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


def edges(window):
    """Splits a window into start and end indices. Will default to have the
    larger padding at the end in case of an odd window.
    """
    start = window // 2
    end = window - start
    return (start, end)

def fast_pad_symmetric(values, window, dtype='f8'):
    """A fast version of numpy n-dimensional symmetric pad.
    In contrast to np.pad, this algorithm only allocates memory once, regardless
    of the number of axes padded. Performance for large data sets is vastly
    improved.
    Note: if the requested padding is 0 along all axes, then this algorithm
    returns the original input ndarray.
    Author: Stian Lode stian.lode@gmail.com
    Args:
        values: n-dimensional ndarray
        window: an iterable of length n
    return:
        a numpy ndarray containing the values with each axis padded according
        to the specified window. The padding is a reflection of the data in
        the input values.
    """
    assert len(values.shape) == len(window)

    if (window <= 0).all():
        return values

    start, end = edges(window)
    new = np.empty(values.shape + window, dtype=dtype)

    slice_stack = []
    for a, b in zip(start, end):
        slice_stack.append(slice(a, None if b == 0 else -b))

    new[tuple(slice_stack)] = values

    slice_stack = []
    for a, b in zip(start, end):
        if a > 0:
            s_to, s_from = slice(a - 1, None, -1), slice(a, 2 * a, None)
            new[tuple(slice_stack + [s_to])] = new[tuple(slice_stack + [s_from])]

        if b > 0:
            e_to, e_from = slice(-1, -b - 1, -1), slice(-2 * b, -b)
            new[tuple(slice_stack + [e_to])] = new[tuple(slice_stack + [e_from])]

        slice_stack.append(slice(None))

    return new

##############################
def subject_cross_validation_split_trials(tc_matrix, label_matrix,target_name, sub_num=None, block_dura=18, n_folds=10, testsize=0.2, valsize=0.1,randomseed=1234):
    ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
    
    Subject_Num, Trial_Num, Region_Num = np.array(tc_matrix).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num>Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 18 ###12s block for MOTOR task

    fmri_data_matrix = []
    label_data_matrix = []
    for subi in range(Subject_Num):
        label_trial_data = np.array(label_matrix[subi])
        condition_mask = pd.Series(label_trial_data).isin(target_name)
        ##condition_mask = pd.Series(label_trial_data).str.split('_', expand=True)[0].isin(target_name)
        fmri_data_matrix.append(tc_matrix[subi][condition_mask, :])
        label_data_matrix.append(label_trial_data[condition_mask])
    fmri_data_matrix = np.array(fmri_data_matrix).astype('float32', casting='same_kind')
    label_data_matrix = np.array(label_data_matrix)
    ##cut the trials into blocks
    chunks = int(np.floor(label_data_matrix.shape[-1] / block_dura))
    fmri_data_block = np.array(np.array_split(fmri_data_matrix, chunks, axis=1)).mean(axis=2).astype('float32',casting='same_kind')
    label_data_block = np.array(np.array_split(label_data_matrix, chunks, axis=1))[:, :, 0]
    print(fmri_data_block.shape,label_data_block.shape)

    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)
    fmri_data_train = np.array([fmri_data_block[:, i, :] for i in train_sid_tmp]).astype('float32', casting='same_kind')
    fmri_data_test = np.array([fmri_data_block[:, i, :] for i in test_sid]).astype('float32', casting='same_kind')
    # print(fmri_data_train.shape,fmri_data_test.shape)

    label_data_train = np.array([label_data_block[:, i] for i in train_sid_tmp])
    label_data_test = np.array([label_data_block[:, i] for i in test_sid])
    # print(label_data_train.shape,label_data_test.shape)

    ###transform the data
    scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test))
    nb_class = len(np.unique(label_data_block))
    Y_test = label_data_test.ravel()
    # print(X_test.shape,Y_test.shape)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        ##preprocess features and labels
        X = np.array(np.vstack([fmri_data_train[i, :, :] for i in train_sid]))
        Y = np.array([label_data_train[i, :] for i in train_sid]).ravel()
        # print(X.shape, Y.shape)
        X_train_scaled.append(scaler.transform(X))
        Y_train_scaled.append(Y)

        X = np.array(np.vstack([fmri_data_train[i, :, :] for i in val_sid]))
        Y = np.array([label_data_train[i, :] for i in val_sid]).ravel()
        # print(X.shape, Y.shape)
        X_val_scaled.append(scaler.transform(X))
        Y_val_scaled.append(Y)

    print('Samples of Subjects for training: %d and testing %d and validating %d with %d classes' % (len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test


def subject_cross_validation_split_trials_new(tc_matrix, label_matrix, target_name, sub_num=None, block_dura=18, flag_event=0,
                                              n_folds=10, train_dataarg=2, testsize=0.2, valsize=0.1, randomseed=1234):
    ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit

    Subject_Num = np.array(tc_matrix).shape[0]
    ##Trial_Num, Region_Num = np.array(tc_matrix[0]).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num > Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 18  ###12s block for MOTOR task
    if not train_dataarg:
        train_dataarg = 1
    train_dataarg = min(train_dataarg, block_dura)

    fmri_data_matrix = []
    label_data_matrix = []
    Trial_dura_pre = 0
    for subi in range(Subject_Num):
        label_trial_data = np.array(label_matrix[subi])
        condition_mask = pd.Series(label_trial_data).isin(target_name)
        ##condition_mask = pd.Series(label_trial_data).str.split('_', expand=True)[0].isin(target_name)

        tc_matrix_select = np.array(tc_matrix[subi][condition_mask, :])
        label_data_select = np.array(label_trial_data[condition_mask])
        ##print(tc_matrix_select.shape,label_data_select.shape)

        le = preprocessing.LabelEncoder()
        le.fit(target_name)
        label_data_int = le.transform(label_data_select)

        ##cut the trials
        label_data_trial_block = np.array(np.split(label_data_select, np.where(np.diff(label_data_int))[0] + 1))
        fmri_data_block = np.array_split(tc_matrix_select, np.where(np.diff(label_data_int))[0] + 1, axis=0)

        trial_duras = [label_data_trial_block[ii].shape[0] for ii in range(label_data_trial_block.shape[0]-1)]
        if len(np.unique(trial_duras))>1 and not flag_event: print("Warning: Using a event design for task ",modality)
        if trial_duras[-1] < block_dura: trial_duras = trial_duras[:-1]
        Trial_dura = min(trial_duras)
        if Trial_dura < 5 and not flag_event:
            print('Warning: Only extract {} TRs for each trial. You need to recheck the event design to make sure!'.format(Trial_dura))
        else:
            if Trial_dura != Trial_dura_pre:
                #print('each trial contains %d volumes/TRs for task %s' % (Trial_dura, modality))
                Trial_dura_pre = Trial_dura

        chunks = int(np.floor(len(label_data_select) / Trial_dura))
        if subi == 1:
            ulabel = [np.unique(x) for x in label_data_trial_block]
            print("After cutting: unique values for each block of trials %s with %d blocks" % (np.array(ulabel), len(ulabel)))
        if label_data_trial_block.shape[0] != chunks:
            try:
                label_data_trial_block = np.array(np.split(label_data_select, chunks))
                fmri_data_block = np.array_split(tc_matrix_select, chunks, axis=0)
            except:
                print("\nWrong cutting of event data...")
                print("Should have %d block-trials but only found %d cuts" % (chunks, label_data_trial_block.shape[0]))
                ulabel = [np.unique(x) for x in label_data_trial_block]
                print("Adjust the cutting: unique values for each block of trials %s with %d blocks\n" % (np.array(ulabel), len(ulabel)))
                chunks = len(trial_duras)

        label_data_trial_block = np.array([label_data_trial_block[i][:Trial_dura] for i in range(chunks)])
        fmri_data_block = np.array([fmri_data_block[i][:Trial_dura, :] for i in range(chunks)])
        if subi == 1: print('first cut:', fmri_data_block.shape, label_data_trial_block.shape)

        #######adjust to event design ??
        ##cut each trial to blocks
        block_dura_used = min(Trial_dura, block_dura)
        chunks = int(np.floor(Trial_dura / block_dura_used))
        if Trial_dura % block_dura_used:
            trial_num_used = Trial_dura // block_dura_used * block_dura_used
            fmri_data_block = np.array(np.vstack(np.array_split(fmri_data_block[:,:trial_num_used,:], chunks, axis=1))).transpose(0,2,1).astype('float32', casting='same_kind')
            label_data_trial_block = np.array(np.vstack(np.array_split(label_data_trial_block[:,:trial_num_used], chunks, axis=1)))[:,0]
        else:
            fmri_data_block = np.array(np.vstack(np.array_split(fmri_data_block, chunks, axis=1))).transpose(0,2,1).astype('float32', casting='same_kind')
            label_data_trial_block = np.array(np.vstack(np.array_split(label_data_trial_block, chunks, axis=1)))[:, 0]
        if subi == 1: print('second cut:', fmri_data_block.shape, label_data_trial_block.shape)
        ##label_data_test = le.transform(label_data_trial_block[:,0]).flatten()
        if subi == 1: print('finalize: reshape data into size:', fmri_data_block.shape, label_data_trial_block.shape)

        fmri_data_matrix.append(fmri_data_block)
        label_data_matrix.append(label_data_trial_block)
    fmri_data_matrix = np.array(fmri_data_matrix) ##.astype('float32', casting='same_kind')
    label_data_matrix = np.array(label_data_matrix)
    print(fmri_data_matrix.shape, label_data_matrix.shape)

    ########spliting into train,val and testing
    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)
    if len(train_sid_tmp)<2 or len(test_sid)<2:
        print("Only %d subjects avaliable. Use all subjects for training and testing" % (sub_num))
        train_sid_tmp = range(sub_num)
        test_sid = range(sub_num)
    fmri_data_train = np.array([fmri_data_matrix[i] for i in train_sid_tmp]) ##.astype('float32', casting='same_kind')
    fmri_data_test = np.array([fmri_data_matrix[i] for i in test_sid]) ##.astype('float32', casting='same_kind')
    print('fmri data for train and test:', fmri_data_train.shape, fmri_data_test.shape)

    label_data_train = np.array([label_data_matrix[i] for i in train_sid_tmp])
    label_data_test = np.array(np.block([label_data_matrix[i] for i in test_sid]))
    print('label data for train and test', label_data_train.shape, label_data_test.shape)

    ###transform the data
    scaler = NDStandardScaler().fit(np.vstack(fmri_data_train))
    ##scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test)) ###.astype('float32', casting='same_kind')
    nb_class = len(target_name)
    Y_test = le.transform(label_data_test) ##.astype('uint8')
    ##print(X_test.shape,Y_test.shape)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        ##preprocess features and labels
        X = np.array(np.vstack([fmri_data_train[i] for i in train_sid]))  ##using vstack or hstack
        Y = np.array(np.block([label_data_train[i] for i in train_sid]))  ##check whether data and label corresponding
        if train_dataarg > 1 and block_dura_used > 1:
            Y = np.repeat(Y, train_dataarg, axis=0).ravel()
            X_new = []
            time_window = np.empty(len(X.shape)).astype(int)
            time_window[-1] = block_dura_used * 2
            for xi in range(X.shape[0]):
                xx = np.array(X[np.random.choice(range(X.shape[0]), size=train_dataarg, replace=True), :, :])
                rand_timeslice = np.random.randint(block_dura_used) ##range(block_dura_used)
                ##xx_wrap = xx.take(range(rand_timeslice,rand_timeslice+block_dura_used), axis=-1, mode='wrap')
                xx_wrap = fast_pad_symmetric(xx, time_window)[:,:,rand_timeslice+block_dura_used:rand_timeslice+2*block_dura_used]
                X_new.append(xx_wrap)
            X = X_new
            X_new = []

        # print('fmri and label data for training:',X.shape, Y.shape)
        X_train_scaled.append(scaler.transform(X))  ##.astype('float32', casting='same_kind')
        Y_train_scaled.append(le.transform(Y))

        X = np.array(np.vstack([fmri_data_train[i] for i in val_sid]))
        Y = np.array(np.block([label_data_train[i] for i in val_sid]))
        # print('fmri and label data for validation:',X.shape, Y.shape)
        X_val_scaled.append(scaler.transform(X))
        Y_val_scaled.append(le.transform(Y))

    print('Samples of Subjects for training: %d and testing %d and validating %d with %d classes' % (
    len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test

def subject_cross_validation_split_trials_eventcut(tc_matrix, label_matrix, target_name, sub_num=None, start_trial=0, hrf_delay=0,
                                                   block_dura=1, sampling=0,flag_event=0,TRstep=1,sub_name=None,
                                                   n_folds=10, train_dataarg=2, testsize=0.2, valsize=0.1, randomseed=123):
    ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit

    Subject_Num = np.array(tc_matrix).shape[0]
    ##Trial_Num, Region_Num = np.array(tc_matrix[0]).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num > Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 1  ###12s block for MOTOR task
    if not train_dataarg:
        train_dataarg = 1
    train_dataarg = min(train_dataarg, block_dura)
    global Trial_dura

    fmri_data_matrix = []
    label_data_matrix = []
    Trial_dura_pre = 0
    for subi in range(Subject_Num):
        label_trial_data = np.array(label_matrix[subi])
        if hrf_delay > 0:
            label_trial_data = np.roll(label_trial_data, -hrf_delay)
        condition_mask = pd.Series(label_trial_data).isin(target_name)
        if start_trial != 0 or hrf_delay > 0:
            ###set the start point of each trial to extract data
            label_trial_data_shift = np.roll(label_trial_data, start_trial)
            condition_mask_shift = pd.Series(label_trial_data_shift).isin(target_name)
            if start_trial < 0:
                condition_mask = np.logical_or(condition_mask_shift, condition_mask)  ##longer: bring forward
                label_trial_data[label_trial_data == 'rest'] = label_trial_data_shift[label_trial_data == 'rest']
            elif start_trial > 0:
                condition_mask = np.logical_and(condition_mask_shift, condition_mask)  ##shorter: postpone

        tc_matrix_select = np.array(tc_matrix[subi][condition_mask, :])
        label_data_select = np.array(label_trial_data[condition_mask])
        ##print(tc_matrix_select.shape,label_data_select.shape)

        le = preprocessing.LabelEncoder()
        le.fit(target_name)
        label_data_int = le.transform(label_data_select)

        ##cut the trials
        label_data_trial_block = np.array(np.split(label_data_select, np.where(np.diff(label_data_int))[0] + 1))
        fmri_data_block = np.array(np.array_split(tc_matrix_select, np.where(np.diff(label_data_int))[0] + 1, axis=0))

        trial_duras = [label_data_trial_block[ii].shape[0] for ii in range(len(label_data_trial_block))]
        ##if len(np.unique(trial_duras)) > 1 and not flag_event : print("Warning: Using a event design for task ", modality)
        if trial_duras[-1] < block_dura:
            #print('Remove the last trial due to too short duration: {} out of requested dura: {}'.format(trial_duras[-1],block_dura))
            trial_duras = trial_duras[:-1]
        try:
            Trial_dura = min(trial_duras)
        except:
            #print(trial_duras)
            continue
        if subi == 1:
            ulabel = [np.unique(x) for x in label_data_trial_block]
            print("After cutting: unique values for each block of trials %s with %d blocks" % (np.array(ulabel), len(ulabel)))

        #######adjust to event design ??
        ##cut each trial to blocks
        fmri_data_block_new = []
        label_data_trial_block_new = []
        for dura,ti in zip(trial_duras,range(len(trial_duras))):
            trial_num_used = min(dura, block_dura)
            xx = fmri_data_block[ti][:trial_num_used, :]
            xx2 = xx.take(range(0, block_dura), axis=0, mode='clip')  ##'warp'
            ##xx2 = fast_pad_symmetric(xx, np.array([xx.shape[0]*2,0]))[xx.shape[0]:xx.shape[0]+block_dura, :]
            fmri_data_block_new.append(np.expand_dims(xx2, axis=0))
            label_data_trial_block_new.append(np.array([label_data_trial_block[ti][0]]))
        label_data_trial_block_new2 = np.concatenate(label_data_trial_block_new,axis=0)
        fmri_data_block_new2 = np.array(np.vstack(fmri_data_block_new)).transpose(0, 2, 1).astype('float32',casting='same_kind')
        fmri_data_block = []; fmri_data_block_new = []
        label_data_trial_block = []; label_data_trial_block_new = []
        ##if subi == 1: print('second cut:', fmri_data_block.shape, label_data_trial_block.shape)
        ##label_data_test = le.transform(label_data_trial_block[:,0]).flatten()
        if subi == 1: print('finalize: reshape data into size:', fmri_data_block_new2.shape, label_data_trial_block_new2.shape)

        fmri_data_matrix.append(fmri_data_block_new2)
        label_data_matrix.append(label_data_trial_block_new2)
    fmri_data_matrix = np.array(fmri_data_matrix)  ##.astype('float32', casting='same_kind')
    label_data_matrix = np.array(label_data_matrix)
    if TRstep > 1:
        fmri_data_matrix = np.array(np.array_split(fmri_data_matrix, TRstep, axis=-1)).mean(axis=0)
    print(fmri_data_matrix.shape, label_data_matrix.shape)

    ################################################################################
    ########spliting into train,val and testing
    sub_num = min(sub_num, label_data_matrix.shape[0])
    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)
    if len(train_sid_tmp)<2 or len(test_sid)<2:
        print("Only %d subjects avaliable. Use all subjects for training and testing" % (sub_num))
        train_sid_tmp = range(sub_num)
        test_sid = range(sub_num)
    if sub_name is not None:
        sub_name_test = [sub_name[i] for i in test_sid]
    else:
        sub_name_test = None
    fmri_data_train = np.array([fmri_data_matrix[i] for i in train_sid_tmp]) ##.astype('float32', casting='same_kind')
    fmri_data_test = np.array([fmri_data_matrix[i] for i in test_sid]) ##.astype('float32', casting='same_kind')
    print('fmri data for train and test:', fmri_data_train.shape, fmri_data_test.shape)

    label_data_train = np.array([label_data_matrix[i] for i in train_sid_tmp])
    print("check test data and labels!")
    ##label_data_test = np.array(([np.expand_dims(label_data_matrix[i],axis=1) for i in test_sid]))
    label_data_test = np.array(([label_data_matrix[i] for i in test_sid]))
    print('label data for train and test', label_data_train.shape, label_data_test.shape)

    ###transform the data
    scaler = NDStandardScaler().fit(np.vstack(fmri_data_train))
    ##scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test)) ###.astype('float32', casting='same_kind')
    nb_class = len(target_name)
    #Y_test = le.transform(np.vstack(label_data_test)).flatten() ##.astype('uint8')
    ##Y_test = le.transform((label_data_test).flatten())
    Y_test = le.transform(np.concatenate(label_data_test).flatten())
    print(X_test.shape,Y_test.shape)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        ##preprocess features and labels
        X = np.array(np.vstack([fmri_data_train[i] for i in train_sid]))  ##using vstack or hstack
        Y = np.array(np.block([label_data_train[i] for i in train_sid]))  ##check whether data and label corresponding

        trial_class_counts = Counter(Y)
        print(trial_class_counts)

        if sampling > 0:
            Y_new = []; X_new = [];
            for trial_class, trial_count in trial_class_counts.items():
                print("Constains {} samples for trial-class {} in the training dataset !".format(trial_count,trial_class))
                tmp_ind = np.where(Y == trial_class)[0]
                if trial_count < int(max(trial_class_counts.values())/2):
                    train_datasample = int(max(trial_class_counts.values())/trial_count)  ##use round
                    Y_new.append(np.repeat(Y[tmp_ind], train_datasample, axis=0).ravel())  ##samples have the same labels

                    if sampling == 1:
                        print("Oversampling training samples for {} times within trial-class {} \n".format(train_datasample,trial_class))
                        time_window = np.empty(len(X.shape)).astype(int)
                        time_window[-1] = block_dura * 2
                        for xi in range(len(tmp_ind)*train_datasample):
                            xx = np.array(X[np.random.choice(tmp_ind, replace=True), :, :])  ##tmp_ind[xi]
                            rand_timeslice = np.random.randint(block_dura)  ##range(block_dura_used)
                            ##xx_wrap = xx.take(range(rand_timeslice,rand_timeslice+block_dura_used), axis=-1, mode='wrap')
                            xx_wrap = fast_pad_symmetric(np.expand_dims(xx, axis=0), time_window)[:, :, rand_timeslice+block_dura:rand_timeslice + 2*block_dura]
                            X_new.append(xx_wrap)
                    elif sampling > 1:
                        print("SMOTE: Generating {} times new samples by averaging among {} neighbours within trial-class {} \n".format(train_datasample, sampling,trial_class))
                        X_new.append(np.array(X[tmp_ind, :, :]))  ##original samples
                        for xi in range(len(tmp_ind)*(train_datasample-1)):
                            ###randomly choose a trial from the list
                            ##xx = np.array(X[np.random.choice(tmp_ind, size=sampling, replace=True), :, :])  ##tmp_ind[xi]
                            ###subject-specific sampling: considering inter-subject variability in brain function
                            subi = np.random.choice(train_sid, replace=True)
                            xx0 = fmri_data_train[subi]
                            yy = label_data_train[subi]
                            tid = np.where(yy == trial_class)[0]
                            xx = np.array(xx0[np.random.choice(tid, size=sampling, replace=True), :, :])
                            ##rand_timeslice = np.random.randint(block_dura_used)
                            ##xx = xx.take(range(rand_timeslice, rand_timeslice + block_dura_used), axis=-1, mode='wrap')
                            X_new.append(np.expand_dims(np.mean(xx, axis=0), axis=0))  ##new samples
                else:
                    X_new.append(X[tmp_ind])
                    Y_new.append(Y[tmp_ind])
            X = np.array(np.vstack(X_new))
            Y = np.array(np.block(Y_new))
            print('After sampling: ', Counter(Y))
        # print('fmri and label data for training:',X.shape, Y.shape)
        X_train_scaled.append(scaler.transform(X))  ##.astype('float32', casting='same_kind')
        Y_train_scaled.append(le.transform(Y))

        X = np.array(np.vstack([fmri_data_train[i] for i in val_sid]))
        Y = np.array(np.block([label_data_train[i] for i in val_sid]))
        # print('fmri and label data for validation:',X.shape, Y.shape)
        X_val_scaled.append(scaler.transform(X))
        Y_val_scaled.append(le.transform(Y))

    print('Samples of Subjects for training: %d and testing %d and validating %d with %d classes' % (len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test, sub_name_test

def subject_cross_validation_split_trials_event(tc_matrix, label_matrix, target_name, sub_num=None, start_trial=0, hrf_delay=0,
                                                block_dura=18, sampling=0,flag_event=0,TRstep=1, sub_name=None,
                                                n_folds=10, train_dataarg=2, drop_rate=0, testsize=0.2, valsize=0.1, randomseed=123):
    ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit

    Subject_Num = np.array(tc_matrix).shape[0]
    ##Trial_Num, Region_Num = np.array(tc_matrix[0]).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num > Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 18  ###12s block for MOTOR task
    if not train_dataarg:
        train_dataarg = 1
    ##train_dataarg = min(train_dataarg, block_dura)
    global Trial_dura

    fmri_data_matrix = []
    label_data_matrix = []
    Trial_dura_pre = 0
    for subi in range(Subject_Num):
        label_trial_data = np.array(label_matrix[subi])
        if hrf_delay > 0:
            label_trial_data = np.roll(label_trial_data, -hrf_delay)
        condition_mask = pd.Series(label_trial_data).isin(target_name)
        if start_trial != 0:
            ###set the start point of each trial to extract data
            label_trial_data_shift = np.roll(label_trial_data, start_trial)
            condition_mask_shift = pd.Series(label_trial_data_shift).isin(target_name)
            if start_trial < 0:
                condition_mask = np.logical_or(condition_mask_shift, condition_mask)  ##longer: bring forward
                label_trial_data[label_trial_data == 'rest'] = label_trial_data_shift[label_trial_data == 'rest']
            elif start_trial > 0:
                condition_mask = np.logical_and(condition_mask_shift, condition_mask)  ##shorter: postpone

        tc_matrix_select = np.array(tc_matrix[subi][condition_mask, :])
        label_data_select = np.array(label_trial_data[condition_mask])
        ##print(tc_matrix_select.shape,label_data_select.shape)

        le = preprocessing.LabelEncoder()
        le.fit(target_name)
        label_data_int = le.transform(label_data_select)

        ##cut the trials
        label_data_trial_block = np.array(np.split(label_data_select, np.where(np.diff(label_data_int))[0] + 1))
        fmri_data_block = np.array(np.array_split(tc_matrix_select, np.where(np.diff(label_data_int))[0] + 1, axis=0))

        trial_duras = [label_data_trial_block[ii].shape[0] for ii in range(len(label_data_trial_block))]
        ###print(trial_duras)
        ##if len(np.unique(trial_duras)) > 1 and not flag_event : print("Warning: Using a event design for task ", modality)
        if trial_duras[-1] < block_dura or trial_duras[-1] < 4:
            #print('Remove the last trial due to too short duration: {} out of requested dura: {}'.format(trial_duras[-1],block_dura))
            trial_duras = trial_duras[:-1]
        try:
            Trial_dura = min(trial_duras)
        except:
            #print(trial_duras)
            continue
        if Trial_dura < 5 and not flag_event:
            print('Warning: Only extract {} TRs for each trial. You need to recheck the event design to make sure!'.format(Trial_dura))
        else:
            if Trial_dura != Trial_dura_pre:
                #print('each trial contains %d volumes/TRs for task %s' % (Trial_dura, modality))
                Trial_dura_pre = Trial_dura
        if subi == 1:
            ulabel = [np.unique(x) for x in label_data_trial_block]
            print("After cutting: unique values for each block of trials %s with %d blocks" % (np.array(ulabel), len(ulabel)))

        #######adjust to event design ??
        ##cut each trial to blocks
        fmri_data_block_new = []
        label_data_trial_block_new = []
        block_dura_used = block_dura #min(Trial_dura, block_dura)
        for dura,ti in zip(trial_duras,range(len(trial_duras))):
            trial_num_used = dura // block_dura_used * block_dura_used
            if trial_num_used < block_dura_used:
                print('\nTask design contains shorter trials:{}! You need to re-consider the block-dura: {} \n'.format(Trial_dura,block_dura_used))
                xx = fmri_data_block[ti][:trial_num_used, :]
                xx2 = xx.take(range(0, block_dura_used), axis=0, mode='clip')  ##'warp'
                ##xx2 = fast_pad_symmetric(xx, np.array([xx.shape[0]*2,0]))[xx.shape[0]:xx.shape[0]+block_dura_used, :]
                fmri_data_block_new.append(np.expand_dims(xx2,axis=0))
                label_data_trial_block_new.append(np.array([label_data_trial_block[ti][0]]))
            else:
                chunks = int(np.floor(trial_num_used // block_dura_used))
                fmri_data_block_new.append(np.array(np.array_split(fmri_data_block[ti][:trial_num_used, :], chunks, axis=0)))
                label_data_trial_block_new.append(np.array(np.repeat(label_data_trial_block[ti][0], chunks)))
        label_data_trial_block_new2 = np.concatenate(label_data_trial_block_new,axis=0)
        try:
            fmri_data_block_new2 = np.array(np.vstack(fmri_data_block_new)).transpose(0, 2, 1).astype('float32',casting='same_kind')
        except:
            print('\nTask design contains some shorter trials:{}! You need to re-consider the block-dura values: {} \n'.format(Trial_dura,block_dura))
            xx = np.array(np.vstack(fmri_data_block_new)).transpose(0, 2, 1)
            fmri_data_block_new2 = xx.take(range(0, block_dura_used), axis=-1, mode='wrap').astype('float32',casting='same_kind')
        fmri_data_block = []; fmri_data_block_new = []
        label_data_trial_block = []; label_data_trial_block_new = []
        ##if subi == 1: print('second cut:', fmri_data_block.shape, label_data_trial_block.shape)
        ##label_data_test = le.transform(label_data_trial_block[:,0]).flatten()
        if subi == 1: print('finalize: reshape data into size:', fmri_data_block_new2.shape, label_data_trial_block_new2.shape)

        fmri_data_matrix.append(fmri_data_block_new2)
        label_data_matrix.append(label_data_trial_block_new2)
    fmri_data_matrix = np.array(fmri_data_matrix)  ##.astype('float32', casting='same_kind')
    label_data_matrix = np.array(label_data_matrix)
    if TRstep > 1:
        fmri_data_matrix = np.array(np.array_split(fmri_data_matrix, TRstep, axis=-1)).mean(axis=0)
    print(fmri_data_matrix.shape, label_data_matrix.shape)

    ################################################################################
    ########spliting into train,val and testing
    sub_num = min(sub_num,label_data_matrix.shape[0])
    print("\nTraining graph convolution using {} subjects! \n".format(sub_num))
    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)
    if len(train_sid_tmp)<2 or len(test_sid)<2:
        print("Only %d subjects avaliable. Use all subjects for training and testing" % (sub_num))
        train_sid_tmp = range(sub_num)
        test_sid = range(sub_num)
    if sub_name is not None:
        sub_name_test = [sub_name[i] for i in test_sid]
    else:
        sub_name_test = None
    fmri_data_train = np.array([fmri_data_matrix[i] for i in train_sid_tmp]) ##.astype('float32', casting='same_kind')
    fmri_data_test = np.array([fmri_data_matrix[i] for i in test_sid]) ##.astype('float32', casting='same_kind')
    print('fmri data for train and test:', fmri_data_train.shape, fmri_data_test.shape)

    label_data_train = np.array([label_data_matrix[i] for i in train_sid_tmp])
    label_data_test = np.array(([np.expand_dims(label_data_matrix[i],axis=1) for i in test_sid]))
    print('label data for train and test', label_data_train.shape, label_data_test.shape)

    ###transform the data
    scaler = NDStandardScaler().fit(np.vstack(fmri_data_train))
    ##scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test)) ###.astype('float32', casting='same_kind')
    nb_class = len(target_name)
    Y_test = le.transform(np.vstack(label_data_test)[:,0]) ##.astype('uint8')
    print(X_test.shape,Y_test.shape)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        ##preprocess features and labels
        X = np.array(np.vstack([fmri_data_train[i] for i in train_sid]))  ##using vstack or hstack
        Y = np.array(np.block([label_data_train[i] for i in train_sid]))  ##check whether data and label corresponding

        #############################################
        if train_dataarg > 1 and drop_rate > 0:
            Y_new = np.repeat(Y, train_dataarg, axis=0).ravel()
            X_new = []
            time_window = np.empty(len(X.shape)).astype(int)
            time_window[-1] = block_dura_used * 2
            for xi in range(X.shape[0] * train_dataarg):
                xx = np.expand_dims(np.array(X[np.random.choice(range(X.shape[0]), replace=True), :, :]),axis=0)
                rand_node = np.random.randint(xx.shape[1],size=int(drop_rate*xx.shape[1]))
                xx[:,rand_node,:] = 1.0 #np.ones((xx.shape[0],len(rand_node),xx.shape[-1]))
                '''
                rand_timeslice = np.random.randint(block_dura_used) ##range(block_dura_used)
                ##xx = xx.take(range(rand_timeslice,rand_timeslice+block_dura_used), axis=-1, mode='wrap')
                xx = fast_pad_symmetric(xx, time_window)[:,:,rand_timeslice+block_dura_used:rand_timeslice+2*block_dura_used]
                '''
                X_new.append(xx)
            X = np.array(np.vstack(X_new))
            Y = np.array(np.block(Y_new))
            X_new = []; Y_new = []

        #########################################sampling
        trial_class_counts = Counter(Y)
        print(trial_class_counts)
        if sampling > 0:
            Y_new = []; X_new = [];
            for trial_class, trial_count in trial_class_counts.items():
                print("Constains {} samples for trial-class {} in the training dataset !".format(trial_count,trial_class))
                tmp_ind = np.where(Y == trial_class)[0]
                if trial_count*2 <= int(max(trial_class_counts.values())):
                    train_datasample = int(max(trial_class_counts.values())/trial_count)  ##use round
                    Y_new.append(np.repeat(Y[tmp_ind], train_datasample, axis=0).ravel())  ##samples have the same labels

                    if sampling == 1:
                        print("Oversampling training samples for {} times within trial-class {} \n".format(train_datasample,trial_class))
                        X_new.append(np.array(X[tmp_ind, :, :]))  ##original samples
                        time_window = np.empty(len(X.shape)).astype(int)
                        time_window[-1] = block_dura_used * 2
                        for xi in range(int(len(tmp_ind)*(train_datasample-1))):
                            xx = np.array(X[np.random.choice(tmp_ind, replace=True), :, :])  ##tmp_ind[xi]
                            rand_timeslice = np.random.randint(block_dura_used)  ##range(block_dura_used)
                            xx_wrap = xx.take(range(rand_timeslice,rand_timeslice+block_dura_used), axis=-1, mode='clip')
                            #xx_wrap = fast_pad_symmetric(np.expand_dims(xx, axis=0), time_window)[:, :, rand_timeslice+block_dura_used:rand_timeslice + 2*block_dura_used]
                            #X_new.append(xx_wrap)
                            X_new.append(np.expand_dims(xx, axis=0))
                    elif sampling > 1:
                        print("SMOTE: Generating {} times new samples by averaging among {} neighbours within trial-class {} \n".format(train_datasample, sampling,trial_class))
                        X_new.append(np.array(X[tmp_ind, :, :]))  ##original samples
                        for xi in range(int(len(tmp_ind)*(train_datasample-1))):
                            ###randomly choose a trial from the list
                            ##xx = np.array(X[np.random.choice(tmp_ind, size=sampling, replace=True), :, :])  ##tmp_ind[xi]
                            ###subject-specific sampling: considering inter-subject variability in brain function
                            tid = []; tnum=0; xx=[]
                            while tnum < sampling:
                                #print("{}:{}".format(len(xx),sampling))
                                subi = np.random.choice(train_sid, replace=True)
                                xx0 = fmri_data_train[subi]
                                yy = label_data_train[subi]
                                tid = np.where(yy == trial_class)[0]
                                if len(tid) > 0:
                                    tnum += len(tid)
                                    xx.append(xx0[tid, :, :])
                            xx = np.vstack(xx)
                            xx = xx[np.random.choice(tnum, size=sampling, replace=True), :, :]
                            ##rand_timeslice = np.random.randint(block_dura_used)
                            ##xx = xx.take(range(rand_timeslice, rand_timeslice + block_dura_used), axis=-1, mode='wrap')
                            X_new.append(np.expand_dims(np.mean(xx, axis=0), axis=0))  ##new samples
                else:
                    X_new.append(X[tmp_ind])
                    Y_new.append(Y[tmp_ind])
            X = np.array(np.vstack(X_new))
            Y = np.array(np.block(Y_new))
            print('After sampling: ', Counter(Y))
        # print('fmri and label data for training:',X.shape, Y.shape)
        X = scaler.transform(X)
        X[np.isnan(X)] = 0; X[np.isinf(X)] = 0
        X_train_scaled.append(X)  ##.astype('float32', casting='same_kind')
        Y_train_scaled.append(le.transform(Y))

        X = np.array(np.vstack([fmri_data_train[i] for i in val_sid]))
        Y = np.array(np.block([label_data_train[i] for i in val_sid]))
        # print('fmri and label data for validation:',X.shape, Y.shape)
        X_val_scaled.append(scaler.transform(X))
        Y_val_scaled.append(le.transform(Y))

    print('Samples of Subjects for training: %d and testing %d and validating %d with %d classes' % (len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test, sub_name_test


def gccn_model_common_param(modality,training_samples,target_name=None,nepochs=50,batch_size=128,layers=3,pool_size=4,hidden_size=256):
    ###common settings for gcn models
    C = len(target_name) + 1
    global block_dura

    gcnn_common = {}
    gcnn_common['dir_name'] = modality + '/' + atlas_name + '_win' + str(block_dura) + '/c'+str(len(target_name))
    if TR_step>1: gcnn_common['dir_name'] += '_step' + str(TR_step) + '/'
    gcnn_common['num_epochs'] = nepochs
    gcnn_common['batch_size'] = batch_size
    gcnn_common['decay_steps'] = training_samples / gcnn_common['batch_size']  ##refine this according to samples
    gcnn_common['eval_frequency'] = int(gcnn_common['num_epochs'] * gcnn_common['decay_steps']/20) ##display 20 lines in total-> eval 20 times
    gcnn_common['brelu'] = 'b2relu' ##'b1relu'
    gcnn_common['pool'] = 'mpool1'
    gcnn_common['initial'] = 'normal'

    ##more params on conv
    # Common hyper-parameters for LeNet5-like networks (two convolutional layers).
    gcnn_common['regularization'] = 5e-4
    gcnn_common['dropout'] = 0.5
    gcnn_common['learning_rate'] = 0.05 #0.02  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
    gcnn_common['decay_rate'] = 0.95
    gcnn_common['momentum'] = 0.9
    gcnn_common['channel'] = block_dura
    gcnn_common['F'] = [32,32,64,64,128,128,128,128] #[32 * math.pow(2, li) for li in
                       # range(layers)]  # [32, 64, 128]  # Number of graph convolutional filters.
    gcnn_common['K'] = [20,10,10,10,5,5,5,5]  #[25 for li in range(layers*2)]  # [25, 25, 25]  # Polynomial orders.
    gcnn_common['p'] = [1,4,1,4,1,4,1,1] #[4,1,4,1,4,1] #[pool_size for li in range(layers)]  # [4, 4, 4]  # Pooling sizes.
    gcnn_common['M'] = [ hidden_size, C]  # Output dimensionality of fully connected layers.
    return gcnn_common

def build_graph_adj_mat(adjacent_mat_file,adjacent_mat_type,coarsening_levels=6,Nneighbours=8, noise_level=0.001):
    ##loading the first-level graph adjacent matrix based on surface neighbourhood
    if adjacent_mat_type.lower() == 'surface':
        print('\n\nLoading adjacent matrix based on counting connected vertices between parcels:',adjacent_mat_file)

        adj_mat = nib.load(adjacent_mat_file).get_data()
        adj_mat = sparse.csr_matrix(adj_mat)
    elif adjacent_mat_type.lower() == 'sc':
        print('\n\nCalculate adjacent graph based on structural covaraince of corrThickness across subjects:',adjacent_mat_file)
        conn_matrix = nib.load(adjacent_mat_file).get_data()

        global mmp_atlas
        atlas_roi = nib.load(mmp_atlas).get_data()
        RegionLabels = [i for i in np.unique(atlas_roi) if i > 0]

        conn_roi_matrix = []
        for li in sorted(RegionLabels):
            tmp_ind = [ind for ind in range(conn_matrix.shape[1]) if atlas_roi[0][ind] == li]
            conn_roi_matrix.append(np.mean(conn_matrix[:, tmp_ind], axis=1))
        conn_roi_matrix = np.transpose(np.array(conn_roi_matrix))
        dist, idx = graph.distance_sklearn_metrics(np.transpose(conn_roi_matrix), k=Nneighbours, metric='cosine')
        adj_mat = graph.adjacency(dist, idx)

    print(adj_mat.shape)
    A = graph.replace_random_edges(adj_mat, noise_level)

    return A, adj_mat

def build_graph_adj_mat_newJune(pathout,mmp_atlas,atlas_name,adjacent_mat_file,graph_type='surf',coarsening_levels=6,noise_level=0.01,Nneighbours=8):
    #####generate brain graphs
    try:
        # import cnn_graph
        ###sys.path.append('/path/to/application/app/folder')
        from cnn_graph.lib_new import graph, coarsening
    except ImportError:
        print('Could not find the package of graph-cnn ...')
        print('Please check the location where cnn_graph is !\n')

    graph_perm_file = os.path.join(pathout, '_'.join([atlas_name, graph_type, 'brain_graph_layer' + str(coarsening_levels) + '_Nei'+str(Nneighbours)+'.pkl']))
    print(graph_perm_file)


    if not os.path.isfile(graph_perm_file):

        if graph_type == 'surf':
            print('\n\nLoading adjacent matrix based on counting connected vertices between parcels:', adjacent_mat_file)
            adj_mat = nib.load(adjacent_mat_file).get_data()
            adj_mat = sparse.csr_matrix(adj_mat)

        elif graph_type == 'SC':
            print('\n\nCalculate adjacent graph based on structural covaraince of corrThickness across subjects:', adjacent_mat_file)
            conn_matrix = nib.load(adjacent_mat_file).get_data()
            Subject_Num, Node_Num = conn_matrix.shape

            atlas_roi = nib.load(mmp_atlas).get_data()
            RegionLabels = [i for i in np.unique(atlas_roi) if i > 0]
            Region_Num = len(RegionLabels)

            tc_matrix_df = pd.DataFrame(data=conn_matrix.ravel(), columns=['tc_signal'])
            tc_matrix_df['roi_label'] = np.repeat(atlas_roi.astype('int'), Subject_Num, axis=0).ravel()
            tc_matrix_df['subj'] = np.repeat(np.arange(Subject_Num).reshape((Subject_Num, 1)), Node_Num, axis=1).ravel()
            # df = pd.DataFrame(values, index=index)

            tc_roi = tc_matrix_df.groupby(['subj', 'roi_label'])
            tc_roi_matrix = tc_roi.mean().values.reshape(Subject_Num, Region_Num)
            #################
            corr_kind = 'partial correlation'   #'correlation' ##
            connectome_measure = connectome.ConnectivityMeasure(kind=corr_kind)
            # connectome_measure = connectome.GroupSparseCovarianceCV()
            # corr_matrix = connectome_measure.fit_transform(np.transpose(subjects_tc_matrix))
            corr_matrix = connectome_measure.fit_transform(np.expand_dims(tc_roi_matrix, axis=0))
            corr_matrix_z = np.tanh(connectome_measure.mean_) ##convert to z-score
            sig = 0.25
            corr_matrix_z = np.exp(corr_matrix_z / sig) ##a Gaussian kernel, defined in Shen 2010

            # k-NN graph.
            idx = np.argsort(-corr_matrix_z)[:, 1:Nneighbours + 1]
            dist = np.array([corr_matrix_z[i, idx[i]] for i in range(corr_matrix_z.shape[0])])
            dist[dist < 1] = 0
            adj_mat = graph.adjacency(dist, idx)

        elif graph_type == 'RSFC':
            subjects_tc_matrix, subname_coding = load_rsfmri_data_matrix(adjacent_mat_file)

            if not os.path.isfile(pathout+atlas_name+'_avg_RSFC_matrix.pkl'):
                corr_kind = 'tangent'
                print('using %s for connectivity measure...' % corr_kind)
                connectome_measure = connectome.ConnectivityMeasure(kind=corr_kind)
                corr_matrix = connectome_measure.fit_transform(np.transpose(subjects_tc_matrix, (0, 2, 1)))
                corr_matrix_z = np.tanh(connectome_measure.mean_)  ##np.mean(np.arctanh(corr_matrix),axis=0)
                with open(pathout+atlas_name+'_avg_RSFC_matrix.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([corr_matrix_z], f)
            else:
                with open(pathout+atlas_name+'_avg_RSFC_matrix.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
                    corr_matrix_z = pickle.load(f)
                corr_matrix_z = corr_matrix_z[0]

            ##sig = 0.01
            sig = np.mean(corr_matrix_z)
            corr_matrix_z = np.exp(corr_matrix_z / sig) ##a Gaussian kernel, defined in Shen 2010
            print(sig, np.histogram(corr_matrix_z, bins=np.arange(10), density=True))

            # k-NN graph.
            idx = np.argsort(-corr_matrix_z)[:, 1:Nneighbours + 1]
            dist = np.array([corr_matrix_z[i, idx[i]] for i in range(corr_matrix_z.shape[0])])
            dist[dist < 1] = 0
            adj_mat = graph.adjacency(dist, idx)

        A = graph.replace_random_edges(adj_mat, noise_level)
        ###build multi-level graph using coarsen (div by 2 at each level)
        graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]
        with open(graph_perm_file, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([A, perm, L], f)
    else:
        # Getting back the objects:
        with open(graph_perm_file, 'rb') as f:  # Python 3: open(..., 'rb')
            A, perm, L = pickle.load(f)

    return A, perm, L

def build_fourier_graph_cnn(gcnn_common,Laplacian_list=None, dropout_lambda=0.0):

    print('\nBuilding convolutional layers with fourier basis of Laplacian\n')
    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#1: two convolutional layers with fourier transform as filters
    dropout_str = '_drop'+str(dropout_lambda) if dropout_lambda>0 else ''
    name = 'fgconv_fgconv_fc_softmax'+dropout_str  # 'Non-Param'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'fourier'
    ###adjust settings for fourier filters
    params['F'] = [32,32,64,64,128,128] #[32 * math.pow(2, li) for li in# range(layers)]  # [32, 64, 128]  # Number of graph convolutional filters.
    params['p'] = [1,4,1,4,1,4] #[4,1,4,1,4,1] #[pool_size for li in range(layers)]  # [4, 4, 4]  # Pooling sizes.
    params['K'] = np.zeros(len(params['p']), dtype=int)

    gcn_node_shapes = [Laplacian_list[li].shape[0] for li in range(len(Laplacian_list))]
    params['K'][0] = int(gcn_node_shapes[0] * (1-dropout_lambda))
    step = 0
    for pi, li in zip(params['p'], range(len(params['p'])-1)):
        if pi == 2:
            params['K'][li+1] = int(gcn_node_shapes[step+1] * (1-dropout_lambda))
            step += 1
        elif pi == 4:
            params['K'][li+1] = int(gcn_node_shapes[step+2] * (1-dropout_lambda))
            step += 2
        elif pi == 1:
            params['K'][li+1] = params['K'][li]
    print(params['K'])
    print(params)

    model = models.cgcnn(config_TF, Laplacian_list, **params)

    print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
    return model, name, params


def build_spline_graph_cnn(gcnn_common, Laplacian_list=None):

    print('\nBuilding convolutional layers with spline basis\n')
    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#2: two convolutional layers with spline basis as filters
    name = 'sgconv_sgconv_fc_softmax'  # 'Non-Param'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'spline'
    print(params)

    model = models.cgcnn(config_TF, Laplacian_list, **params)
    print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
    return model, name, params

def build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=None, flag_firstorder=0):


    print('\nBuilding convolutional layers with Chebyshev polynomial\n')
    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#3: two convolutional layers with Chebyshev polynomial as filters
    name = 'cgconv_cgconv_fc_softmax'  # 'Non-Param'
    if flag_firstorder:  name += '_firstorder'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'chebyshev5'

    params['learning_rate'] = 0.001  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
    params['decay_rate'] = 0.9  ##0.95
    params['initial'] = 'he'

    ####adjust param setting for chebyshev
    if not flag_firstorder:
        params['F'] = [32,32,64,64,128,128] #[32 * math.pow(2, li) for li in range(layers)]  # [32, 64, 128]  # Number of graph convolutional filters.
        params['p'] = [1,4,1,4,1,4] #[4,1,4,1,4,1] #[pool_size for li in range(layers)]  # [4, 4, 4]  # Pooling sizes.
        ###params['K'] = [1 for li in range(len(gcnn_common['p']))]  # [25, 25, 25]  # Polynomial orders.
        params['K'] = [20, 10, 10, 10, 5, 5]  # [25 for li in range(layers*2)]  # [25, 25, 25]  # Polynomial orders.
    else:
        params['F'] = [32,32,32,32,32,32] #[32 * math.pow(2, li) for li in range(layers)]  # [32, 64, 128]  # Number of graph convolutional filters.
        params['p'] = [1,1,1,1,1,1] #[4,1,4,1,4,1] #[pool_size for li in range(layers)]  # [4, 4, 4]  # Pooling sizes.
        params['K'] = [1 for li in range(len(params['p']))]  # [25, 25, 25]  # Polynomial orders.
        ##params['M'] = gcnn_common['M'][-1:]  ##remove the last dense layer
    params['M'] = [gcnn_common['M'][0]*2,] +gcnn_common['M']
    print(params)

    model = models.cgcnn(config_TF,Laplacian_list, **params)
    print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
    return model, name, params

def show_gcn_results(s, fontsize=None):
    if fontsize:
        plt.rc('pdf', fonttype=42)
        plt.rc('ps', fonttype=42)
        plt.rc('font', size=fontsize)         # controls default text sizes
        plt.rc('axes', titlesize=fontsize)    # fontsize of the axes title
        plt.rc('axes', labelsize=fontsize)    # fontsize of the x any y labels
        plt.rc('xtick', labelsize=fontsize)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=fontsize)   # fontsize of the tick labels
        plt.rc('legend', fontsize=fontsize)   # legend fontsize
        plt.rc('figure', titlesize=fontsize)  # size of the figure title
    print('  accuracy        F1             loss        time [ms]  name')
    print('test  train   test  train   test     train')
    for name in sorted(s.names):
        print('{:5.2f} {:5.2f}   {:5.2f} {:5.2f}   {:.2e} {:.2e}   {:3.0f}   {}'.format(
                s.test_accuracy[name], s.train_accuracy[name],
                s.test_f1[name], s.train_f1[name],
                s.test_loss[name], s.train_loss[name], s.fit_time[name]*1000, name))
    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for name in sorted(s.names):
        steps = np.arange(len(s.fit_accuracies[name])) + 1
        steps *= s.params[name]['eval_frequency']
        ax[0].plot(steps, s.fit_accuracies[name], '.-', label=name)
        ax[1].plot(steps, s.fit_losses[name], '.-', label=name)
    ax[0].set_xlim(min(steps), max(steps))
    ax[1].set_xlim(min(steps), max(steps))
    ax[0].set_xlabel('step')
    ax[1].set_xlabel('step')
    ax[0].set_ylabel('validation accuracy')
    ax[1].set_ylabel('training loss')
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='upper right')
    #fig.savefig('training.pdf')
    '''
    print('')
    return s


def build_graph_cnn_subject_predict(subjects_tc_matrix, subjects_trial_label_matrix, target_name, modality,checkpoint_dir,
                                    block_dura=1, flag_sampling=0, flag_event=0, start_trial=0, flag_cut_events=False, flag_starttr=False, hrf_delay=0,
                                    layers=3,pool_size=4,hidden_size=256,nepochs=100,
                                    testsize=0.2, valsize=0.2, batch_size=128, sub_num=None,sub_name=None):
    ###classification using graph convolution neural networks with subject-specific split of train, val and test
    Subject_Num = np.array(subjects_tc_matrix).shape[0]
    Trial_Num, Region_Num = np.array(subjects_tc_matrix[0]).shape
    if Trial_Num != np.array(subjects_trial_label_matrix[0]).shape[0]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
        print('Adjust subject numbers for event data')
        print('Need to run preclean_data before to ensure size matching between fmri and event data!')
        Subject_Num = min(np.array(subjects_tc_matrix).shape[0], np.array(subjects_trial_label_matrix).shape[0])
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)

    ##split data into train, val and test in subject-level
    if not sub_num or sub_num > Subject_Num:
        sub_num = Subject_Num
    else:
        if not sub_name:
            sub_name = sub_name[:sub_num]

    if not flag_cut_events:
        X_train, Y_train, X_val, Y_val, X_test, Y_test,sub_name_test = \
            subject_cross_validation_split_trials_event(subjects_tc_matrix, subjects_trial_label_matrix, target_name, start_trial=start_trial, hrf_delay=hrf_delay,
                                                        train_dataarg=dataarg, drop_rate=droprate, sampling=flag_sampling, flag_event=flag_event,
                                                        block_dura=block_dura, n_folds=1, testsize=testsize, valsize=valsize,
                                                        sub_num=sub_num,sub_name=sub_name)
    else:
        print("Only extracting first {} TRs from each trial".format(block_dura))
        X_train, Y_train, X_val, Y_val, X_test, Y_test,sub_name_test = \
            subject_cross_validation_split_trials_eventcut(subjects_tc_matrix, subjects_trial_label_matrix, target_name, start_trial=start_trial, hrf_delay=hrf_delay,
                                                           train_dataarg=dataarg, sampling=flag_sampling, flag_event=flag_event,
                                                           block_dura=block_dura, n_folds=1, testsize=testsize, valsize=valsize,
                                                           sub_num=sub_num,sub_name=sub_name)

    global Trial_dura
    trial_dura = int((Trial_dura - start_trial)/block_dura)
    if trial_dura<1: trial_dura = 1
    print('\nEvaluating on {} subjects with {} dura of trials\n'.format(len(sub_name_test),trial_dura))
    ##################################################################
    ###prepare for gcn model
    ###pre-setting of common parameters
    global adj_mat_file, adj_mat_type, coarsening_levels
    if flag_event:
        modality_str = modality + '_ev'
    else:
        modality_str = modality
    gcnn_common = gccn_model_common_param(modality_str, X_train[0].shape[0], target_name,
                                          layers=layers, pool_size=pool_size, hidden_size=hidden_size, batch_size=batch_size,
                                          nepochs=nepochs)
    model_perf = utils.model_perf()

    '''
    if adj_mat_type.lower() == 'surf':
        adj_mat_file = pathout + args.atlas_name+'_adjacency_mat_white.pconn.nii'
    elif adj_mat_type.lower() == 'sc':
        print('Only avaliable for MMP atlas for sc calculation now!')
        mmp_atlas = pathdata + "codes/HCP_S1200_GroupAvg_v1/"+"Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
        adj_mat_file = pathdata + 'codes/HCP_S1200_GroupAvg_v1/S1200.All.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'

    graph_perm_file =pathout + args.atlas_name+'_surf_brain_graph_layer'+ str(coarsening_levels) + '_N512.pkl'
        
    if not os.path.isfile(graph_perm_file):
        A, adj_mat = build_graph_adj_mat(adj_mat_file, adj_mat_type)
        ###build multi-level graph using coarsen (div by 2 at each level)
        graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]

        with open(graph_perm_file, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([A, perm, L], f)
    else:
        # Getting back the objects:
        with open(graph_perm_file, 'rb') as f:  # Python 3: open(..., 'rb')
            A, perm, L = pickle.load(f)
    '''
    ##load brain graphs
    A, perm, L = build_graph_adj_mat_newJune(pathout, mmp_atlas, atlas_name, adj_mat_file, graph_type=adj_mat_type,coarsening_levels=coarsening_levels)

    from collections import namedtuple
    Record = namedtuple("gcnn_name", ["gcnn_model", "gcnn_params"])
    ##s = {"test_id1": Record("res1", "time1"), "test_id2": Record("res2", "time2")}
    ##s["test_id1"].resultValue

    ###cut the order of graph fourier transform
    #model1, gcnn_name1, params1 = build_fourier_graph_cnn(gcnn_common, Laplacian_list=L, dropout_lambda=0)

    model8, gcnn_name8, params8 = build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=L, flag_firstorder=1)
    #model9, gcnn_name9, params9 = build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=L, flag_firstorder=0)

    gcnn_model_dicts = {##gcnn_name1: Record(model1, params1),
                        gcnn_name8: Record(model8, params8),
                        ##gcnn_name9: Record(model9, params9),
                        }
    ##initalization
    train_acc = {}
    test_acc = {}
    val_acc = {}
    for name in gcnn_model_dicts.keys():
        train_acc[name] = []
        test_acc[name] = []
        val_acc[name] = []

    ###subject-specific cross-validation
    for name in gcnn_model_dicts.keys():
        print('\n\nTraining graph cnn using %s filters!' % name)
        ###training
        model = gcnn_model_dicts[name].gcnn_model
        params = gcnn_model_dicts[name].gcnn_params
        print(name, params)

        ckp_path = Path(os.path.join(checkpoint_dir,atlas_name+'_win' + str(block_dura), 'c'+str(len(target_name))+name))
        for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(len(X_train))):
            train_data = coarsening.perm_data_3d(x_train, perm)
            train_labels = y_train  # np.array([d[x] for x in y_train])
            val_data = coarsening.perm_data_3d(x_val, perm)
            val_labels = y_val  # np.array([d[x] for x in y_val])
            test_data = coarsening.perm_data_3d(X_test, perm)
            test_labels = Y_test
            print('\nFold #%d: training on %d samples with %d features and %d channels, validating on %d samples and testing on %d samples' %
                  (tcount + 1, train_data.shape[0], train_data.shape[1], block_dura, val_data.shape[0], test_data.shape[0]))

            start_time = time.time()
            ##evaluation
            print('Evaluating on Testing set with test  accuracy')
            ###test_logits, test_pred, test_loss = model_perf.predict_allmodel(ckp_path, test_data, test_labels,target_name=target_name, batch_size=batch_size)
            test_logits, test_pred, test_loss, acc = model_perf.predict(ckp_path, test_data, test_labels, target_name=target_name,batch_size=batch_size,trial_dura=trial_dura, flag_starttr=flag_starttr,sub_name=sub_name_test)
            test_acc[name].append(acc)
            print('Evaluating on Training set with train  accuracy')
            train_logits, train_pred, train_loss, acc = model_perf.predict(ckp_path, train_data, train_labels, target_name=target_name, batch_size=batch_size,trial_dura=trial_dura, flag_starttr=flag_starttr)
            train_acc[name].append(acc)
            #val_logits, val_pred, val_loss, acc = model_perf.predict(ckp_path, val_data, val_labels, target_name=target_name,batch_size=batch_size)
            print("Finish model tranning in {} s".format(time.time() - start_time))

    if flag_starttr:
        method_name_dict = {'cgconv_cgconv_fc_softmax': 'ChebNet',
                            'cgconv_cgconv_fc_softmax_firstorder': '1stChebNet',
                            'fgconv_fgconv_fc_softmax': 'Spectral'}
        ##start_tr = range(start_trial, Trial_dura)
        start_tr = range(int((Trial_dura - start_trial)/block_dura))
        result = pd.DataFrame()
        for model_name in gcnn_model_dicts.keys():
            result_df = pd.DataFrame(np.array(test_acc[model_name]).squeeze().transpose(), columns=target_name)
            result_df['block_dura'] = block_dura
            result_df['start_tr'] = start_tr
            result_df['model'] = method_name_dict[model_name]
            result = pd.concat([result, result_df], ignore_index=True )

        result_filename = os.path.join('train_log/','_'.join(('result_predict',modality,'tasks_start_trial',str(block_dura)+'block','testc'+str(len(target_name))+'.csv')))
        if os.path.isfile(result_filename):
            print("Result file already exist:", result_filename)
            xx = os.path.basename(result_filename)
            xx_new = xx.split('.')[0] + "_new"
            result_filename = os.path.join('train_log/', '.'.join((xx_new, 'csv')))
        print("\nSave the dataframe to csv file:",result_filename)
        result.to_csv(result_filename, sep='\t', encoding='utf-8', index=False)

    return test_acc, train_acc

def build_graph_cnn_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,target_name,modality,
                                       block_dura=1,flag_sampling=0,flag_event=0,start_trial=0,flag_cut_events=False,hrf_delay=0,
                                       layers=3,pool_size=4,hidden_size=256,batch_size=128,nepochs=100,TRstep=1,
                                       flag_multi_gcn_compare=0, my_cv_fold=10,testsize=0.2,valsize=0.2,sub_num=None,sub_name=None):
    ###classification using graph convolution neural networks with subject-specific split of train, val and test
    Subject_Num = np.array(subjects_tc_matrix).shape[0]
    Trial_Num, Region_Num = np.array(subjects_tc_matrix[0]).shape
    if Trial_Num != np.array(subjects_trial_label_matrix[0]).shape[0]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
        print('Adjust subject numbers for event data')
        print('Need to run preclean_data before to ensure size matching between fmri and event data!')
        Subject_Num = min(np.array(subjects_tc_matrix).shape[0],np.array(subjects_trial_label_matrix).shape[0])
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)


    ##split data into train, val and test in subject-level
    if not sub_num or sub_num > Subject_Num:
        sub_num = Subject_Num
    else:
        if not sub_name:
            sub_name = sub_name[:sub_num]
    if not flag_cut_events:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, tmp = \
            subject_cross_validation_split_trials_event(subjects_tc_matrix, subjects_trial_label_matrix, target_name, start_trial=start_trial,hrf_delay=hrf_delay,
                                                        train_dataarg=dataarg,drop_rate=droprate,sampling=flag_sampling,flag_event=flag_event,TRstep=TRstep,
                                                        block_dura=block_dura,n_folds=my_cv_fold, testsize=testsize, valsize=valsize, sub_num=sub_num)
    else:
        print("Only extracting first {} TRs from each trial".format(block_dura))
        X_train, Y_train, X_val, Y_val, X_test, Y_test, tmp= \
            subject_cross_validation_split_trials_eventcut(subjects_tc_matrix, subjects_trial_label_matrix, target_name, start_trial=start_trial,hrf_delay=hrf_delay,
                                                           train_dataarg=dataarg,sampling=flag_sampling,flag_event=flag_event,TRstep=TRstep,
                                                           block_dura=block_dura,n_folds=my_cv_fold, testsize=testsize, valsize=valsize, sub_num=sub_num)

    #X_train_all = np.array(np.vstack((X_train[0], X_val[0])))
    #Y_train_all = np.array(np.concatenate((Y_train[0], Y_val[0]), axis=0))
    #print('sample size for training and testing: ', X_train_all.shape, Y_train_all.shape)

    ##################################################################
    ###prepare for gcn model
    ###pre-setting of common parameters
    global adj_mat_file,adj_mat_type, coarsening_levels
    if flag_event:
        modality_str = modality + '_ev'
    else:
        modality_str = modality
    gcnn_common = gccn_model_common_param(modality_str,X_train[0].shape[0],target_name,
                                          layers=layers,pool_size=pool_size, hidden_size=hidden_size,batch_size=batch_size,
                                          nepochs=nepochs)
    model_perf = utils.model_perf()

    '''
    if adj_mat_type.lower() == 'surf':
        adj_mat_file = pathout + args.atlas_name+'_adjacency_mat_white.pconn.nii'
    elif adj_mat_type.lower() == 'sc':
        print('Only avaliable for MMP atlas for sc calculation now!')
        mmp_atlas = pathdata + "codes/HCP_S1200_GroupAvg_v1/"+"Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
        adj_mat_file = pathdata + 'codes/HCP_S1200_GroupAvg_v1/S1200.All.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'

    graph_perm_file =pathout + args.atlas_name+'_surf_brain_graph_layer'+ str(coarsening_levels) + '_N512.pkl'

    if not os.path.isfile(graph_perm_file):
        A, adj_mat = build_graph_adj_mat(adj_mat_file, adj_mat_type)
        ###build multi-level graph using coarsen (div by 2 at each level)
        graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]

        with open(graph_perm_file, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([A, perm, L], f)
    else:
        # Getting back the objects:
        with open(graph_perm_file, 'rb') as f:  # Python 3: open(..., 'rb')
            A, perm, L = pickle.load(f)
    '''

    ##load brain graphs
    A, perm, L = build_graph_adj_mat_newJune(pathout, mmp_atlas, atlas_name, adj_mat_file, graph_type=adj_mat_type, coarsening_levels=coarsening_levels)

    if flag_multi_gcn_compare:
        from collections import namedtuple
        Record = namedtuple("gcnn_name", ["gcnn_model", "gcnn_params"])
        ##s = {"test_id1": Record("res1", "time1"), "test_id2": Record("res2", "time2")}
        ##s["test_id1"].resultValue

        ###cut the order of graph fourier transform
        #model1, gcnn_name1, params1 = build_fourier_graph_cnn(gcnn_common,Laplacian_list=L, dropout_lambda=0)
        #model2, gcnn_name2, params2 = build_fourier_graph_cnn(gcnn_common, Laplacian_list=L, dropout_lambda=0.2)
        #model3, gcnn_name3, params3 = build_fourier_graph_cnn(gcnn_common, Laplacian_list=L, dropout_lambda=0.4)
        #model4, gcnn_name4, params4 = build_fourier_graph_cnn(gcnn_common, Laplacian_list=L, dropout_lambda=0.6)
        #model5, gcnn_name5, params5 = build_fourier_graph_cnn(gcnn_common, Laplacian_list=L, dropout_lambda=0.8)

        model8, gcnn_name8, params8 = build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=L,flag_firstorder=1)
        #model9, gcnn_name9, params9 = build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=L,flag_firstorder=0)

        #model3, gcnn_name3, params3 = build_spline_graph_cnn(gcnn_common,Laplacian_list=L)
        #model4, gcnn_name4, params4 = build_chebyshev_graph_cnn(gcnn_common,Laplacian_list=L)
        gcnn_model_dicts = {#gcnn_name1: Record(model1,params1),
                            #gcnn_name2: Record(model2,params2),
                            #gcnn_name3: Record(model3,params3),
                            #gcnn_name4: Record(model4,params4),
                            #gcnn_name5: Record(model5, params5),

                            gcnn_name8: Record(model8, params8),
                            #gcnn_name9: Record(model9, params9),
                            }
        ##initalization
        train_acc = {};
        train_loss = {};
        test_acc = {};
        test_loss = {};
        val_acc = {};
        val_loss = {};
        for name in gcnn_model_dicts.keys():
            train_acc[name] = []
            train_loss[name] = []
            test_acc[name] = []
            test_loss[name] = []
            val_acc[name] = []
            val_loss[name] = []

        ###subject-specific cross-validation
        test_labels = Y_test
        for name in gcnn_model_dicts.keys():
            print('\n\nTraining graph cnn using %s filters!' % name)
            ###training
            model = gcnn_model_dicts[name].gcnn_model
            params = gcnn_model_dicts[name].gcnn_params
            print(name, params)

            accuracy=[]; loss=[];  t_step=[];
            for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(len(X_train))):
                train_data = coarsening.perm_data_3d(x_train, perm)
                train_labels = y_train #np.array([d[x] for x in y_train])
                val_data = coarsening.perm_data_3d(x_val, perm)
                val_labels = y_val #np.array([d[x] for x in y_val])
                test_data = coarsening.perm_data_3d(X_test, perm)
                test_labels = Y_test
                print('\nFold #%d: training on %d samples with %d features and %d channels, validating on %d samples and testing on %d samples' %
                    (tcount + 1, train_data.shape[0], train_data.shape[1], block_dura, val_data.shape[0], test_data.shape[0]))

                #acc, los, tstep = model.fit(train_data, train_labels, val_data, val_labels)
                #accuracy.append(acc)
                #loss.append(los)
                #t_step.append(tstep)

                start_time = time.time()
                ##evaluation
                model_perf.test(model, name, params,
                                train_data, train_labels, val_data, val_labels, test_data, test_labels, target_name=target_name)
                train_acc[name].append(model_perf.train_accuracy[name])
                train_loss[name].append(model_perf.train_loss[name])
                test_acc[name].append(model_perf.test_accuracy[name])
                test_loss[name].append(model_perf.test_loss[name])
                val_acc[name].append(model_perf.fit_accuracies[name])
                val_loss[name].append(model_perf.fit_losses[name])
                print("Finish model tranning in {} s".format(time.time()-start_time))

            print('\nResults for graph-cnn using %s filters!' % name)
            print('Accuracy of training:{},testing:{}'.format(np.mean(train_acc[name]), np.mean(test_acc[name])))
            print('Accuracy of validation:mean=%2f' % np.mean(np.max(val_acc[name], axis=1)))

    else:
        ##model#1: two convolutional layers with fourier transform as filters
        model, name, params = build_fourier_graph_cnn(gcnn_common, Laplacian_list=L)

        d = {k: v + 1 for v, k in enumerate(sorted(set(Y_test)))}
        test_labels = np.array([d[x] for x in Y_test])
        print(np.unique(Y_test))

        train_acc = [];
        train_loss = [];
        test_acc = [];
        test_loss = [];
        val_acc = [];
        val_loss = [];
        accuracy = [];
        loss = [];
        t_step = [];
        for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(2)):
            train_data = coarsening.perm_data_3d(x_train, perm)
            train_labels = np.array([d[x] for x in y_train])
            val_data = coarsening.perm_data_3d(x_val, perm)
            val_labels = np.array([d[x] for x in y_val])
            test_data = coarsening.perm_data_3d(X_test, perm)
            print('\nFold #%d: training on %d samples with %d features, validating on %d samples and testing on %d samples'
                  % (tcount + 1, train_data.shape[0], train_data.shape[1], val_data.shape[0], test_data.shape[0]))

            ###training
            #model = models.cgcnn(config_TF, L, **params)
            acc, los, tstep = model.fit(train_data, train_labels, val_data, val_labels)
            accuracy.append(acc)
            loss.append(los)
            t_step.append(tstep)

            ##evaluation
            model_perf.test(model, name, params,
                            train_data, train_labels, val_data, val_labels, test_data, test_labels)
            train_acc.append(model_perf.train_accuracy[name])
            train_loss.append(model_perf.train_loss[name])
            test_acc.append(model_perf.test_accuracy[name])
            test_loss.append(model_perf.test_loss[name])
            val_acc.append(model_perf.fit_accuracies[name])
            val_loss.append(model_perf.fit_losses[name])
            print('\n')

    return train_acc, test_acc, val_acc, model_perf

##################################
####main function
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The description of the parameters')

    parser.add_argument('--atlas_name', '-a', default='MMP', help="(required, str, default='MMP') Choosing which atlas to map fmri data ", type=str)
    parser.add_argument('--adj_type', '-j', default='surf', help="(required, str, default='surf') Choosing which method to build the adjacent matrix ", type=str)
    parser.add_argument('--task_modality', '-m', default='wm', help="(required, str, default='wm') Choosing which modality of fmri data for modeling", type=str)
    parser.add_argument('--dataarg', '-d', default=1, help='(optional, int, default=1) #Samples generated in data-argument for training set', type=int)
    parser.add_argument('--droprate', '-o', default=0.1, help='(optional, float, default=0.1) #Dropout rate on graph nodes',type=float)
    parser.add_argument('--kfold', '-k', default=1, help='(optional, float, default=1) The potion of subjects used in validation set',type=int)
    parser.add_argument('--block_dura', '-b', default=1, help='(optional, int, default=1) The duration of fmri volumes in each data sample', type=int)
    parser.add_argument('--coarsening_levels', '-c', default=6, help='(optional, int, default=6) Coarsening_levels from the original graph', type=int)
    parser.add_argument('--subnum_test', '-s', default=0, help='(optional, int, default=0) The number of subjects to be used ', type=int)
    parser.add_argument('--flag_eventdesign', '-g', default=False, help='(optional, bool, default=False) Using event design instead of block design ', type=bool)
    parser.add_argument('--flag_sampling', '-p', default=0, help='(optional, int, default=0) Sampling scheme used for unbalanced categories ',type=int)
    parser.add_argument('--flag_cut_event', '-l', default=False, help='(optional, bool, default=False) Cutting the first few TRs instead of stacking blocks ', type=bool)
    parser.add_argument('--start_trial', '-f', default=0, help='(optional, int, default=0) Starting time of fMRI data and events ',type=int)
    parser.add_argument('--hrf_delay', default=0, help='(optional, int, default=0) Shifting event design for several TRs due to hrf delay ',type=int)

    parser.add_argument('--learning_rate', '-r', default=0.001, help="(required, float, default=0.001) Choosing to run 2d-cnn or 3d-cnn model", type=float)
    parser.add_argument('--val_size', '-v', default=0.1, help='(optional, float, default=0.1) The potion of subjects used in validation set', type=float)
    parser.add_argument('--test_size', '-t', default=0.2, help='(optional, float, default=0.2) The potion of subjects used in testing set', type=float)
    parser.add_argument('--batch_size', '-i', default=128, help='(optional, int, default=128) The batch size for model training', type=int)
    parser.add_argument('--nepochs', '-e', default=100, help='(optional, int, default=100) The number of epochs for model training', type=int)
    parser.add_argument('--hidden_size', default=256, help='(optional, int, default=100) number of units for the last fully connected layer', type=int)

    parser.add_argument('--predict_only', '-q', default=False, help='(optional, bool, default=False) The number of epochs for model training', type=bool)
    parser.add_argument('--flag_starttr', '-u', default=False,help='(optional, bool, default=False) start from the cue phase', type=bool)
    parser.add_argument('--TR_step', default=1, help='(optional, int, default=1) Sampling steps of TR for fMRI time series',type=int)

    args = parser.parse_args()

    block_dura = args.block_dura
    dataarg = args.dataarg
    droprate = args.droprate
    coarsening_levels = args.coarsening_levels
    adj_mat_type = args.adj_type
    TR_step = args.TR_step

    with tf.device("/cpu:0"):
        ##########pre-setting of the task and data enviornment
        task_contrasts, modality = bulid_dict_task_modularity(args.task_modality)
        target_name = np.unique(list(task_contrasts.values()))
        print(target_name,len(target_name))

        if args.atlas_name == 'MMP':
            atlas_name = 'MMP'
            mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
        elif args.atlas_name == 'BNA':
            atlas_name = 'BNA'
            mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "BN_Atlas_246_2mm.nii.gz"
        elif args.atlas_name == 'CBIG1k':
            atlas_name = 'CBIG1k'
            mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"
        elif args.atlas_name == 'CBIG400':
            atlas_name = 'CBIG400'
            mmp_atlas = pathout + "../codes/HCP_S1200_GroupAvg_v1/" + "Schaefer2018_400Parcels_17Networks_order.dlabel.nii"

        if args.adj_type == 'surf':
            adj_mat_file = pathout + atlas_name + '_adjacency_mat_white.pconn.nii'
        elif args.adj_type == 'SC':
            adj_mat_file = pathatlas + 'S1200.All.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'
        elif args.adj_type == 'RSFC':
            adj_mat_file = os.path.join(pathout, '_'.join(('rsfmri', 'REST1', 'LR', atlas_name, "ROI_act_1200R_test2_March2019_new.lmdb")))

        lmdb_filename = pathout + modality + "_"+args.atlas_name+"_ROI_act_1200R_test_Dec2018_ALL.lmdb" ##Dec2018_ALL
        print("Mapping fmri data to atlas: ", args.atlas_name)

        checkpoint_dir = "cnn_graph_test/checkpoints/" + modality + "/" + args.atlas_name + "_win" + str(block_dura) + "/"
        ##shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        ##########loading fmri and event files
        fmri_files, confound_files, subjects = load_fmri_data(pathdata,modality)
        #print('including %d fmri files and %d confounds files \n\n' % (len(fmri_files), len(confound_files)))

        ev_filename = "_event_labels_1200R_test_Dec2018_ALL_new2.h5"
        if modality == 'MOTOR' or 'WM' in modality or modality == 'ALLTasks':
            ev_filename = "_event_labels_1200R_test_Dec2018_ALL_newLR.h5"
        subjects_trial_label_matrix, sub_name_ev, Trial_dura = load_event_files(pathdata, modality, fmri_files,confound_files,ev_filename=ev_filename,
                                                                                flag_event=args.flag_eventdesign)
        if block_dura == 1:
            print('\n use each trial as one sample during model training ')

        #print('each trial contains %d volumes/TRs for task %s' % (Trial_dura,modality))
        #print('Collecting event design files for subjects and saved into matrix ...\n' , np.array(subjects_trial_label_matrix).shape)

        print('Collecting fmri data from lmdb file...')
        subjects_tc_matrix, subname_coding = load_fmri_data_from_lmdb(lmdb_filename)
        print(np.array(subjects_tc_matrix).shape)
        print('\n')

        #####
        subjects_tc_matrix, subjects_trial_label_matrix, fmri_sub_name = preclean_data_for_shape_match_new(subjects_tc_matrix, subjects_trial_label_matrix,
                                                                                            subname_coding, sub_name_ev)
        Subject_Num = np.array(subjects_tc_matrix).shape[0]
        Region_Num = np.array(subjects_tc_matrix).shape[-1]
        print(np.array(subjects_trial_label_matrix).shape)
        print(np.array(subjects_tc_matrix).shape)
        #print(np.unique(subjects_trial_label_matrix))


    if args.subnum_test > 0 and args.subnum_test < Subject_Num:
        subnum_test = args.subnum_test
        fmri_tc_data = subjects_tc_matrix[:subnum_test]
        label_data = subjects_trial_label_matrix[:subnum_test]
    else:
        fmri_tc_data = subjects_tc_matrix
        label_data = subjects_trial_label_matrix
    ##################################################################
    ###prepare for gcn model
    if not args.predict_only:
        train_acc, test_acc, val_acc, model_perf = \
            build_graph_cnn_subject_validation(fmri_tc_data, label_data, target_name, modality,
                                               testsize=args.test_size, valsize=args.val_size, my_cv_fold=args.kfold,
                                               batch_size=args.batch_size,nepochs=args.nepochs,TRstep=TR_step,
                                               block_dura=block_dura, #nepochs=10,batch_size=4,my_cv_fold=2,
                                               flag_sampling=args.flag_sampling, flag_event=args.flag_eventdesign,
                                               start_trial=args.start_trial,flag_cut_events=args.flag_cut_event, hrf_delay=args.hrf_delay,
                                               flag_multi_gcn_compare=1)
        print(train_acc, test_acc, val_acc)
        for gcnn_name in train_acc.keys():
            print('GCN model:',gcnn_name)
            print('Accuracy of training:{},val:{}, testing:{}'.
                  format(np.mean(train_acc[gcnn_name]), np.mean(np.max(val_acc[gcnn_name], axis=1)), np.mean(test_acc[gcnn_name])))
        ###summarize the results
        #model_perf.show()
        ss = show_gcn_results(model_perf)

    else:
        checkpoint_dir = "cnn_graph_test/checkpoints/" + modality + "/"
        start_trial = args.start_trial
        print("Using the first {}TRs from each trial starting at {} TR".format(block_dura,start_trial))
        build_graph_cnn_subject_predict(fmri_tc_data, label_data, target_name, modality, checkpoint_dir,
                                      testsize=args.test_size, valsize=args.val_size,
                                      block_dura=block_dura, batch_size=args.batch_size,
                                      flag_sampling=args.flag_sampling, flag_event=args.flag_eventdesign,sub_name=fmri_sub_name,
                                      start_trial=start_trial,hrf_delay=args.hrf_delay,
                                      flag_cut_events=args.flag_cut_event,flag_starttr=args.flag_starttr)



    ###to run
    ## mod='MOTOR'
    # blocks=1
    # python HCP_task_fmri_gcn_test7.py --task_modality=${mod} --block_dura=${blocks} --flag_cut_event=True --predict_only=True --start_trial=17

    sys.exit(0)