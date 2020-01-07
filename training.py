#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###fMRI decoding: using event signals instead of activation pattern from glm


import os
import numpy as np
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from keras.utils import np_utils

from lib_new import coarsening
import lib_new.models_gcn as models
from utils import *
from model import *
from configure_fmri import *

print('Finish Loading packages!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The description of the parameters')

    parser.add_argument('--task_modality', '-m', default='motor', help="(required, str, default='wm') Choosing which modality of fmri data for modeling", type=str)
    parser.add_argument('--block_dura', '-b', default=1, help='(optional, int, default=1) The duration of fmri volumes in each data sample', type=int)

    parser.add_argument('--eigorder', '-e', default=0, help='(optional, int, default=1) Order of Laplacian eigenvectors', type=int)
    parser.add_argument('--Korder', '-k', default=10, help='(optional, int, default=1) Order of Chebychev polynomials', type=int)
    args = parser.parse_args()

    block_dura = args.block_dura
    task_modality = args.task_modality

    subjects_tc_matrix, label_matrix, modality, target_name, fmri_sub_name = prepare_fmri_data(pathdata, task_modality, pathout, atlas_name=atlas_name, verbose=1)
    Nlabels = len(target_name) + 1

    fmri_data_matrix, label_data_matrix, fmri_sub_name, Trial_dura = matching_fmri_data_to_trials_event(subjects_tc_matrix, label_matrix, target_name, fmri_sub_name, block_dura=block_dura)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, testset_subjects = subject_split_trials_event(fmri_data_matrix, label_data_matrix, fmri_sub_name, target_name, block_dura=block_dura)

    print('\nStep 6: Model training started!')
    gcnn_common = gccn_model_common_param(modality, len(Y_train), target_name,block_dura=block_dura)
    model_perf = models.model_perf()

    ##load brain graphs
    A, perm, L = build_graph_adj_mat_newJune(pathout, mmp_atlas, atlas_name, adj_mat_file, graph_type=adj_mat_type, coarsening_levels=coarsening_levels)

    from collections import namedtuple
    Record = namedtuple("gcnn_name", ["gcnn_model", "gcnn_params"])
    ###cut the order of graph fourier transform
    model1, gcnn_name1, params1 = build_fourier_graph_cnn(gcnn_common,Laplacian_list=L, eigorders=args.eigorder)
    model8, gcnn_name8, params8 = build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=L, Korder=args.Korder, flag_firstorder=0)
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
