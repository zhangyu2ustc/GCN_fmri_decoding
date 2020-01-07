#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###fMRI decoding: using event signals instead of activation pattern from glm

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

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from keras.utils import np_utils

import pickle
import lmdb
import tensorflow as tf
from tensorpack.utils.serialize import dumps, loads
print(tf.__version__)

from lib_new import coarsening
import lib_new.models_gcn as models
from model import *
from configure_fmri import *

##print('Finish Loading packages!')


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


###################################################
def bulid_dict_task_modularity(modality):
    ##build the dict for different subtypes of events under different modalities
    motor_task_con = {"rf": "footR_mot",
                      "lf": "footL_mot",
                      "rh": "handR_mot",
                      "lh": "handL_mot",
                      "t": "tongue_mot"}
    '''
    lang_task_con =  {"present_math":  "math_lang",
                      "question_math": "math_lang",
                      "response_math": "math_lang",
                      "present_story":  "story_lang",
                      "question_story": "story_lang" ,
                      "response_story": "story_lang"}
    '''
    lang_task_con = {"math": "math_lang",
                     "story": "story_lang"}
    emotion_task_con = {"fear": "fear_emo",
                        "neut": "non_emo"}
    gambl_task_con = {"win_event": "win_gamb",
                      "loss_event": "loss_gamb",
                      "neut_event": "non_gamb"}
    reson_task_con = {"match": "match_reson",
                      "relation": "relat_reson"}
    social_task_con = {"mental": "mental_soc",
                       "rnd": "random_soc"}
    wm_task_con = {"2bk_body": "body2b_wm",
                   "2bk_faces": "face2b_wm",
                   "2bk_places": "place2b_wm",
                   "2bk_tools": "tool2b_wm",
                   "0bk_body": "body0b_wm",
                   "0bk_faces": "face0b_wm",
                   "0bk_places": "place0b_wm",
                   "0bk_tools": "tool0b_wm"}

    dicts = [motor_task_con, lang_task_con, emotion_task_con, reson_task_con, social_task_con, wm_task_con] ##gambl_task_con,
    from collections import defaultdict
    all_task_con = defaultdict(list)  # uses set to avoid duplicates
    for d in dicts:
        for k, v in d.items():
            all_task_con[k].append(v)  ## all_task_con[k]=v to remove the list []

    mod_chosen = modality[:3].lower().strip()
    mod_choices = {'mot': 'MOTOR',
                   'lan': 'LANGUAGE',
                   'emo': 'EMOTION',
                   'gam': 'GAMBLING',
                   'rel': 'RELATIONAL',
                   'soc': 'SOCIAL',
                   'wm': 'WM',
                   'all': 'ALLTasks'}
    task_choices = {'mot': motor_task_con,
                    'lan': lang_task_con,
                    'emo': emotion_task_con,
                    'gam': gambl_task_con,
                    'rel': reson_task_con,
                    'soc': social_task_con,
                    'wm': wm_task_con,
                    'all': all_task_con}

    modality = mod_choices.get(mod_chosen, 'default')
    task_contrasts = task_choices.get(mod_chosen, 'default')
    return task_contrasts, modality

def load_rsfmri_data_matrix(lmdb_filename,Trial_Num=1200):
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


def load_fmri_data_from_lmdb(lmdb_filename,modality='MOTOR'):
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

def load_event_from_h5(events_all_subjects_file, task_contrasts, Trial_Num=284, TR=0.72,verbose=0):
    ################################
    ###loading all event designs from h5

    if not os.path.isfile(events_all_subjects_file):
        print("event file not exist!:",events_all_subjects_file)
        return None

    subjects_trial_labels = pd.read_csv(events_all_subjects_file, sep="\t", encoding="utf8")
    ###print(subjects_trial_labels.keys())

    try:
        label_matrix = subjects_trial_labels['label_data'].values
        # print(label_matrix[0],label_matrix[1])
        # xx = label_matrix[0].split(",")
        subjects_trial_label_matrix = []
        for subi in range(len(label_matrix)):
            xx = [x.replace("['", "").replace("']", "") for x in label_matrix[subi].split("', '")]
            subjects_trial_label_matrix.append(xx)
        subjects_trial_label_matrix = pd.DataFrame(data=(subjects_trial_label_matrix))
        Trial_Num = subjects_trial_label_matrix.shape[-1]
    except:
        Trial_Num = sum(subjects_trial_labels.columns.str.contains('trial[0-9]'))
        print('only extracting {} trials from event design'.format(Trial_Num))
        subjects_trial_label_matrix = subjects_trial_labels.loc[:, subjects_trial_labels.columns.str.contains('trial[0-9]')]

    ##subjects_trial_label_matrix = subjects_trial_labels.values.tolist()
    trialID = subjects_trial_labels['trialID']
    sub_name = subjects_trial_labels['subject'].tolist()
    coding_direct = subjects_trial_labels['coding']
    if verbose:
        trial_class_counts = Counter(list(subjects_trial_label_matrix.iloc[0, :]))
        print(trial_class_counts)
        print('Collecting trial info from file:', events_all_subjects_file)
        print(np.array(subjects_trial_label_matrix).shape, len(sub_name), len(np.unique(sub_name)),len(coding_direct))

    return subjects_trial_label_matrix, sub_name


def preclean_data_for_shape_match_new(subjects_tc_matrix,subjects_trial_label_matrix, fmri_sub_name, ev_sub_name):
    print("\nPre-clean the fmri and event data to make sure the matching shapes between two arrays!")
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


def prepare_fmri_data(pathdata, task_modality, pathout, atlas_name='MMP', test_size=0.2,val_size=0.1, test_sub_num=0,randseed=1234,verbose=0):

    task_contrasts, modality = bulid_dict_task_modularity(task_modality)
    target_name = np.unique(list(task_contrasts.values()))
    if verbose:
        print(target_name, len(target_name))

    lmdb_filename = pathout + modality + "_" + atlas_name+"_ROI_act_1200R_test_Dec2018_ALL.lmdb" ##Dec2018_ALL
    print("Mapping fmri data to atlas: ", atlas_name)
    subjects_tc_matrix, subname_coding = load_fmri_data_from_lmdb(lmdb_filename)
    Trial_Num, Region_Num = subjects_tc_matrix[0].shape
    if verbose:
        print("\nStep1: collect fmri data from lmdb file by mapping fmri signals onto {} atlas with {} regions".format(atlas_name, Region_Num))
        print('%d subjects included in the dataset' % len(subjects_tc_matrix))
        print("one example:", subname_coding[0])

    ev_filename = "_event_labels_1200R_test_Dec2018_ALL_new2.h5"
    if modality == 'MOTOR' or modality == 'WM' or modality == 'ALLTasks':
        ev_filename = "_event_labels_1200R_test_Dec2018_ALL_newLR.h5"
    events_all_subjects_file = pathout + modality + ev_filename
    label_matrix, ev_sub_name = load_event_from_h5(events_all_subjects_file,task_contrasts)
    if verbose:
        print("\nStep2: Loading all event designs for {} task".format(modality))
        print('including %d subjects of event designs' % len(subjects_tc_matrix))
        print("one example:", ev_sub_name[0])

    ###match fmri-files with event-design matrix
    subjects_tc_matrix, label_matrix, fmri_sub_name = preclean_data_for_shape_match_new(subjects_tc_matrix, label_matrix, subname_coding, ev_sub_name)
    if verbose:
        print("\nStep3: Matching {} fMRI data with {} event files for all subjects".format(len(subjects_tc_matrix),len(label_matrix)))
        ##print(fmri_sub_name)

    return subjects_tc_matrix, label_matrix, modality, target_name, fmri_sub_name


def matching_fmri_data_to_trials_event(tc_matrix, label_matrix, target_name, fmri_sub_name, block_dura=15, start_trial=0, hrf_delay=0,
                                       flag_event=0, TRstep=1,verbose=1):

    #####matching time-series to event design
    if verbose:
        print("\nStep4: Matching fMRI volumes with event designs for each task trial of {} conditions".format(len(target_name)))

    fmri_data_matrix = []
    label_data_matrix = []
    Trial_dura_pre = 0; Trial_dura = 0
    for subi in range(len(label_matrix)):
        label_trial_data = np.array(label_matrix[subi])
        if hrf_delay > 0:
            ##considering the delay of hrf by right shifting event design
            label_trial_data = np.roll(label_trial_data, hrf_delay)
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

        if sum(np.where(condition_mask)[0]) < 1: fmri_sub_name[subi] = []
        ###only extracting the selected task conditions
        tc_matrix_select = np.array(tc_matrix[subi][condition_mask, :])
        label_data_select = np.array(label_trial_data[condition_mask])
        ##print(tc_matrix_select.shape,label_data_select.shape)

        le = preprocessing.LabelEncoder()
        le.fit(target_name)
        label_data_int = le.transform(label_data_select)

        ##cut the trials according to block design of trials
        label_data_trial_block = np.array(np.split(label_data_select, np.where(np.diff(label_data_int))[0] + 1))
        fmri_data_block = np.array(np.array_split(tc_matrix_select, np.where(np.diff(label_data_int))[0] + 1, axis=0))

        ## extract the duration of each trial
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


        ##cut each trial to variable time windows
        fmri_data_block_new = []
        label_data_trial_block_new = []
        block_dura_used = block_dura #min(Trial_dura, block_dura)
        for dura,ti in zip(trial_duras,range(len(trial_duras))):
            trial_num_used = dura // block_dura_used * block_dura_used
            if trial_num_used < block_dura_used:
                #print('\nTask design contains shorter trials:{}! You need to re-consider the block-dura: {} \n'.format(Trial_dura,block_dura_used))
                if trial_num_used < 1: trial_num_used = dura
                xx = fmri_data_block[ti][:trial_num_used, :]
                xx2 = xx.take(range(0, block_dura_used), axis=0, mode='clip')  ##'warp'
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
            print('\nTask design contains some shorter trials:{}! You need to re-consider the block-dura values: {}'.format(Trial_dura,block_dura))
            xx = np.array(np.vstack(fmri_data_block_new)).transpose(0, 2, 1)
            fmri_data_block_new2 = xx.take(range(0, block_dura_used), axis=-1, mode='clip').astype('float32',casting='same_kind')
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
    print("fmri/event data shape after matching trials:", fmri_data_matrix.shape, label_data_matrix.shape)
    fmri_sub_name = list(filter(None, fmri_sub_name))

    return fmri_data_matrix, label_data_matrix, fmri_sub_name, Trial_dura

def scan_split_trials_event(tc_matrix, label_matrix, fmri_sub_name, target_name, block_dura=15, sub_num=None, sampling=0,
                               test_size=0.2, val_size=0.1, randomseed=123, verbose=1):
    ################################################################################
    ########spliting into train,val and testing
    Subject_Num = len(tc_matrix)
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num > Subject_Num:
        test_sub_num = Subject_Num
    else:
        test_sub_num = sub_num

    subjects = [sub.split("_")[0] for sub in fmri_sub_name]
    print("Remaining {} functional scans of {} subjects".format(len(subjects), len(np.unique(subjects))))

    np.random.seed(randseed)
    subjectList = np.random.permutation(range(test_sub_num))
    subjectTest = int(test_size*test_sub_num)
    train_sid_tmp = subjectList[:test_sub_num-subjectTest]
    test_sid = subjectList[-subjectTest:]
    testset_subjects = subjects[test_sid]
    ##validation and training sets
    subjectList = np.random.permutation(train_sid_tmp)
    subjectVal = int(val_size * test_sub_num)
    train_sid = subjectList[: test_sub_num-subjectVal-subjectTest]
    val_sid = subjectList[-subjectVal:]

    ##test set
    fmri_data_test = np.array([tc_matrix[i] for i in test_sid])
    label_data_test = [label_matrix[i] for i in test_sid]
    ##print(fmri_data_test.shape, np.array(label_data_test).shape )
    ##train
    fmri_data_train = np.array([tc_matrix[i] for i in train_sid])
    label_data_train = [label_matrix[i] for i in train_sid]
    ##val
    fmri_data_val = np.array([tc_matrix[i] for i in val_sid])
    label_data_val = [label_matrix[i] for i in val_sid]
    ##print(fmri_data_train.shape, np.array(label_data_train).shape)

    if verbose:
        print('\nStep 5: Training the model {} fmri scans and validated on {} fmri scans'
              .format(len(train_sid), len(val_sid)))
        print('training on {} subjects, validated on {} subjects and testing on {} subjects'
              .format(len(train_sid), len(val_sid), len(test_sid)))
        print("in total of {} subjects and {} functional scans".format(len(np.unique(subjects)), len(subjects)))
    #############################################
    ###transform the data
    scaler = NDStandardScaler().fit(np.vstack(fmri_data_train))
    ##scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test)) ###.astype('float32', casting='same_kind')
    nb_class = len(target_name)
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    Y_test = le.transform(np.block(label_data_test)) ##.astype('uint8')
    print("data shape for test set:", X_test.shape,Y_test.shape)

    ##preprocess features and labels
    train_sid = np.random.permutation(range(len(label_data_train)))
    X = np.array(np.vstack([fmri_data_train[i] for i in train_sid]))   ##using vstack or hstack
    Y = np.array(np.block([label_data_train[i] for i in train_sid]))  ##check whether data and label corresponding
    #########################################sampling
    trial_class_counts = Counter(Y)
    print(trial_class_counts)

    ###for the unbalanced classes, using different sampling stradige
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
                    time_window[-1] = block_dura * 2
                    for xi in range(int(len(tmp_ind)*(train_datasample-1))):
                        xx = np.array(X[np.random.choice(tmp_ind, replace=True), :, :])  ##tmp_ind[xi]
                        rand_timeslice = np.random.randint(block_dura)  ##range(block_dura_used)
                        xx_wrap = xx.take(range(rand_timeslice,rand_timeslice+block_dura), axis=-1, mode='clip')
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
    X_train_scaled = np.array(X)  ##.astype('float32', casting='same_kind')
    Y_train_scaled = le.transform(Y)
    print("data shape for training set:", X_train_scaled.shape,Y_train_scaled.shape)

    val_sid = np.random.permutation(range(len(label_data_val)))
    X = np.array(np.vstack([fmri_data_val[i] for i in val_sid]))
    Y = np.array(np.block([label_data_val[i] for i in val_sid]))
    # print('fmri and label data for validation:',X.shape, Y.shape)
    X_val_scaled = np.array(scaler.transform(X))
    Y_val_scaled = le.transform(Y)
    print("data shape for validation set:", X_val_scaled.shape,Y_val_scaled.shape)

    print('Samples size for training: %d and testing %d and validating %d with %d classes' % (len(Y_train_scaled), len(Y_test), len(Y_val_scaled), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test, testset_subjects


def subject_split_trials_event(tc_matrix, label_matrix, fmri_sub_name, target_name, block_dura=15, sub_num=None, sampling=0,
                               test_size=0.2, val_size=0.1, randomseed=123, verbose=1):
    ################################################################################
    ########spliting into train,val and testing
    Subject_Num = len(tc_matrix)
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num > Subject_Num:
        sub_num = Subject_Num

    subjects = [sub.split("_")[0] for sub in fmri_sub_name]
    subjects_unique = np.unique(subjects)
    test_sub_num = len(subjects_unique)
    print("Remaining {} functional scans of {} subjects".format(len(subjects), len(np.unique(subjects))))

    np.random.seed(randseed)
    subjectList = np.random.permutation(range(test_sub_num))
    subjectTest = int(test_size*test_sub_num)
    train_sid_tmp = subjectList[:test_sub_num-subjectTest]
    test_sid = subjectList[-subjectTest:]
    testset_subjects = subjects_unique[test_sid]

    ##convert from subject index to file index
    ##test set
    test_file_sid = [si for si,sub in enumerate(subjects) if sub in subjects_unique[test_sid]]
    print(np.array(tc_matrix).shape,len(label_matrix))
    fmri_data_test = np.array([tc_matrix[i] for i in test_file_sid])
    label_data_test = [label_matrix[i] for i in test_file_sid]
    ##print(fmri_data_test.shape, np.array(label_data_test).shape )

    ##validation and training sets
    subjectList = np.random.permutation(train_sid_tmp)
    subjectVal = int(val_size * test_sub_num)
    train_sid = subjectList[: test_sub_num-subjectVal-subjectTest]
    val_sid = subjectList[-subjectVal:]

    ##train
    train_file_sid = [si for si, sub in enumerate(subjects) if sub in subjects_unique[train_sid]]
    fmri_data_train = np.array([tc_matrix[i] for i in train_file_sid])
    label_data_train = [label_matrix[i] for i in train_file_sid]
    ##val
    val_file_sid = [si for si, sub in enumerate(subjects) if sub in subjects_unique[val_sid]]
    fmri_data_val = np.array([tc_matrix[i] for i in val_file_sid])
    label_data_val = [label_matrix[i] for i in val_file_sid]
    ##print(fmri_data_train.shape, np.array(label_data_train).shape)

    if verbose:
        print('\nStep 5: Training the model {} fmri scans and validated on {} fmri scans'
              .format(len(train_file_sid), len(val_file_sid)))
        print('training on {} subjects, validated on {} subjects and testing on {} subjects'
              .format(len(train_sid), len(val_sid), len(test_sid)))
        print("in total of {} subjects and {} functional scans".format(len(np.unique(subjects)), len(subjects)))
    #############################################
    ###transform the data
    scaler = NDStandardScaler().fit(np.vstack(fmri_data_train))
    ##scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test)) ###.astype('float32', casting='same_kind')
    nb_class = len(target_name)
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    Y_test = le.transform(np.block(label_data_test)) ##.astype('uint8')
    print("data shape for test set:", X_test.shape,Y_test.shape)

    ##preprocess features and labels
    train_sid = np.random.permutation(range(len(label_data_train)))
    X = np.array(np.vstack([fmri_data_train[i] for i in train_sid]))   ##using vstack or hstack
    Y = np.array(np.block([label_data_train[i] for i in train_sid]))  ##check whether data and label corresponding
    #########################################sampling
    trial_class_counts = Counter(Y)
    print(trial_class_counts)

    ###for the unbalanced classes, using different sampling stradige
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
                    time_window[-1] = block_dura * 2
                    for xi in range(int(len(tmp_ind)*(train_datasample-1))):
                        xx = np.array(X[np.random.choice(tmp_ind, replace=True), :, :])  ##tmp_ind[xi]
                        rand_timeslice = np.random.randint(block_dura)  ##range(block_dura_used)
                        xx_wrap = xx.take(range(rand_timeslice,rand_timeslice+block_dura), axis=-1, mode='clip')
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
    X_train_scaled = np.array(X)  ##.astype('float32', casting='same_kind')
    Y_train_scaled = le.transform(Y)
    print("data shape for training set:", X_train_scaled.shape,Y_train_scaled.shape)

    val_sid = np.random.permutation(range(len(label_data_val)))
    X = np.array(np.vstack([fmri_data_val[i] for i in val_sid]))
    Y = np.array(np.block([label_data_val[i] for i in val_sid]))
    # print('fmri and label data for validation:',X.shape, Y.shape)
    X_val_scaled = np.array(scaler.transform(X))
    Y_val_scaled = le.transform(Y)
    print("data shape for validation set:", X_val_scaled.shape,Y_val_scaled.shape)

    print('Samples size for training: %d and testing %d and validating %d with %d classes' % (len(Y_train_scaled), len(Y_test), len(Y_val_scaled), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test, testset_subjects

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The description of the parameters')
    parser.add_argument('--task_modality', '-m', default='motor', help="(required, str, default='wm') Choosing which modality of fmri data for modeling", type=str)
    parser.add_argument('--block_dura', '-b', default=1, help='(optional, int, default=1) The duration of fmri volumes in each data sample', type=int)
    args = parser.parse_args()

    block_dura = args.block_dura
    task_modality = args.task_modality

    subjects_tc_matrix, label_matrix, modality, target_name, fmri_sub_name = prepare_fmri_data(pathdata, task_modality, pathout, atlas_name=atlas_name, verbose=1)
    Nlabels = len(target_name) + 1

    fmri_data_matrix, label_data_matrix, fmri_sub_name, Trial_dura = matching_fmri_data_to_trials_event(subjects_tc_matrix, label_matrix, target_name, fmri_sub_name, block_dura=block_dura)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = subject_split_trials_event(fmri_data_matrix, label_data_matrix, fmri_sub_name, target_name, block_dura=block_dura)

    print('\nStep 6: Model training started!')
    gcnn_common = gccn_model_common_param(modality, len(Y_train), target_name ,block_dura=block_dura)
    model_perf = models.model_perf()

    ##load brain graphs
    A, perm, L = build_graph_adj_mat_newJune(pathout, mmp_atlas, atlas_name, adj_mat_file, graph_type=adj_mat_type, coarsening_levels=coarsening_levels)

    from collections import namedtuple
    Record = namedtuple("gcnn_name", ["gcnn_model", "gcnn_params"])
    ###cut the order of graph fourier transform
    model1, gcnn_name1, params1 = build_fourier_graph_cnn(gcnn_common,Laplacian_list=L, eigorders=10)
    model8, gcnn_name8, params8 = build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=L, Korder=10, flag_firstorder=0)
    model9, gcnn_name9, params9 = build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=L,flag_firstorder=1)

    gcnn_model_dicts = {gcnn_name1: Record(model1,params1),
                        gcnn_name8: Record(model8, params8),
                        gcnn_name9: Record(model9, params9),
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

        train_data = coarsening.perm_data_3d(X_train, perm)
        train_labels = Y_train #np.array([d[x] for x in y_train])
        val_data = coarsening.perm_data_3d(X_val, perm)
        val_labels = Y_val #np.array([d[x] for x in y_val])
        test_data = coarsening.perm_data_3d(X_test, perm)
        test_labels = Y_test

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
'''