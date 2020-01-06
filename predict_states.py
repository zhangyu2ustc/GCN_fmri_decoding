#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###fMRI decoding: using event signals instead of activation pattern from glm


import os
import numpy as np
import pandas as pd
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
    parser.add_argument('--start_trial', '-s', default=0, help='(optional, int, default=1) Starting time point of fMRI data and events', type=int)
    parser.add_argument('--cal_acc_curve', '-u', default=False,help='(optional, bool, default=False) Calculating the accuarcy curve as a function of time-elapsed-from-onset', type=bool)

    args = parser.parse_args()

    block_dura = args.block_dura
    task_modality = args.task_modality
    start_trial = args.start_trial
    flag_starttr = args.cal_acc_curve

    subjects_tc_matrix, label_matrix, modality, target_name, fmri_sub_name = prepare_fmri_data(pathdata, task_modality, pathout, atlas_name=atlas_name, verbose=1)
    Nlabels = len(target_name) + 1

    fmri_data_matrix, label_data_matrix, fmri_sub_name,Trial_dura = matching_fmri_data_to_trials_event(subjects_tc_matrix, label_matrix, target_name, fmri_sub_name, block_dura=block_dura)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, testset_subjects = subject_split_trials_event(fmri_data_matrix, label_data_matrix, fmri_sub_name, target_name, block_dura=block_dura)

    print('\nPredicting cognitive state using trained model')
    checkpoint_dir = "checkpoints/" + modality + "/"
    print("Loading trained model from checkpoint folder:", checkpoint_dir)

    #########################################################
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

            ckp_path = Path(os.path.join(checkpoint_dir, atlas_name+'_win' + str(block_dura), 'c'+str(len(target_name))+name))

            train_data = coarsening.perm_data_3d(X_train, perm)
            train_labels = Y_train  # np.array([d[x] for x in y_train])
            val_data = coarsening.perm_data_3d(X_val, perm)
            val_labels = Y_val  # np.array([d[x] for x in y_val])
            test_data = coarsening.perm_data_3d(X_test, perm)
            test_labels = Y_test

            start_time = time.time()
            ##evaluation
            print('Evaluating on Testing set with test  accuracy')
            ###test_logits, test_pred, test_loss = model_perf.predict_allmodel(ckp_path, test_data, test_labels,target_name=target_name, batch_size=batch_size)
            test_logits, test_pred, test_loss, acc = model_perf.predict(ckp_path, test_data, test_labels, target_name=target_name,
                                                                        batch_size=batch_size,trial_dura=Trial_dura, flag_starttr=flag_starttr,
                                                                        sub_name=testset_subjects)
            test_acc[name].append(acc)
            print('Evaluating on Training set with train  accuracy')
            train_logits, train_pred, train_loss, acc = model_perf.predict(ckp_path, train_data, train_labels, target_name=target_name,
                                                                           batch_size=batch_size,trial_dura=Trial_dura, flag_starttr=flag_starttr)
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

            result_filename = os.path.join('train_logs/','_'.join(('result_predict',modality,'tasks_start_trial',str(block_dura)+'block','testc'+str(len(target_name))+'.csv')))
            if os.path.isfile(result_filename):
                print("Result file already exist:", result_filename)
                xx = os.path.basename(result_filename)
                xx_new = xx.split('.')[0] + "_new"
                result_filename = os.path.join('train_logs/', '.'.join((xx_new, 'csv')))
            print("\nSave the dataframe to csv file:",result_filename)
            result.to_csv(result_filename, sep='\t', encoding='utf-8', index=False)


        print('\nResults for graph-cnn using %s filters!' % name)
        print('Accuracy of training:{},testing:{}'.format(np.mean(train_acc[name]), np.mean(test_acc[name])))
        print('Accuracy of validation:mean=%2f' % np.mean(np.max(val_acc[name], axis=1)))
