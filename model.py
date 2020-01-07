#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###fMRI decoding: using event signals instead of activation pattern from glm

import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse
from nilearn import connectome

import tensorflow.keras.backend as K
import tensorflow as tf

from lib_new import graph, coarsening
import lib_new.models_gcn as models
from configure_fmri import *


print('\nAvaliable GPUs for usage: %d \n' % num_GPU)
config_TF = tf.ConfigProto(intra_op_parallelism_threads=num_CPU,\
        inter_op_parallelism_threads=num_CPU, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config_TF)
tf.keras.backend.set_session(session)


#####build different neural networks
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
            from utils import load_rsfmri_data_matrix
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


def gccn_model_common_param(modality,training_samples,target_name=None,block_dura=15, eval_report=20,
                            nepochs=100,batch_size=128,layers=6,pool_size=1,hidden_size=256,):
    ###common settings for gcn models
    C = len(target_name) + 1

    gcnn_common = {}
    gcnn_common['dir_name'] = modality + '/' + atlas_name + '_win' + str(block_dura) + '/c'+str(len(target_name))
    if TR_step>1: gcnn_common['dir_name'] += '_step' + str(TR_step) + '/'
    gcnn_common['num_epochs'] = nepochs
    gcnn_common['batch_size'] = batch_size
    gcnn_common['decay_steps'] = training_samples / gcnn_common['batch_size']  ##refine this according to samples
    gcnn_common['eval_frequency'] = int(gcnn_common['num_epochs'] * gcnn_common['decay_steps']/eval_report) ##display 20 lines in total-> eval 20 times
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


def build_fourier_graph_cnn(gcnn_common,Laplacian_list=None, dropout_lambda=0.0, eigorders=10):

    print('\nBuilding convolutional layers with fourier basis of Laplacian\n')
    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#1: two convolutional layers with fourier transform as filters
    dropout_str = 'drop'+str(dropout_lambda) if dropout_lambda>0 else ''
    eigorder_str = 'full' if not eigorders else 'K'+str(eigorders)
    name = '_'.join(('fgconv_fgconv_fc_softmax',eigorder_str, dropout_str))
    if eigorders==10: name = 'fgconv_fgconv_fc_softmax'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'fourier'
    ###adjust settings for fourier filters
    params['F'] = [32,32,32,32,32,32] #[32,32,64,64,128,128] #[32 * math.pow(2, li) for li in# range(layers)]  # [32, 64, 128]  # Number of graph convolutional filters.
    params['p'] = [1,1,1,1,1,1] #[1,4,1,4,1,4] #[4,1,4,1,4,1] #[pool_size for li in range(layers)]  # [4, 4, 4]  # Pooling sizes.
    params['M'] = [gcnn_common['M'][0] * 2, ] + gcnn_common['M']

    if not eigorders:
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
    else:
        params['K'] = [eigorders, eigorders, eigorders, eigorders, eigorders, eigorders]

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

def build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=None, Korder=5, flag_firstorder=0):

    print('\nBuilding convolutional layers with Chebyshev polynomial\n')
    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#3: two convolutional layers with Chebyshev polynomial as filters
    name = 'cgconv_cgconv_fc_softmax' + '_K'+str(Korder) # 'Non-Param'
    if flag_firstorder:  name = 'cgconv_cgconv_fc_softmax' + '_firstorder'
    if Korder==10: name = 'cgconv_cgconv_fc_softmax'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'chebyshev5'

    params['learning_rate'] = 0.001  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
    params['decay_rate'] = 0.9  ##0.95
    params['initial'] = 'he'

    ####adjust param setting for chebyshev
    if not flag_firstorder:
        params['F'] = [32,32,32,32,32,32] #[32,32,64,64,128,128] #[32 * math.pow(2, li) for li in range(layers)]  # [32, 64, 128]  # Number of graph convolutional filters.
        params['p'] = [1,1,1,1,1,1] #[1,4,1,4,1,4] #[4,1,4,1,4,1] #[pool_size for li in range(layers)]  # [4, 4, 4]  # Pooling sizes.
        ###params['K'] = [1 for li in range(len(gcnn_common['p']))]  # [25, 25, 25]  # Polynomial orders.
        params['K'] = [Korder,Korder,Korder,Korder,Korder,Korder] #[20, 10, 10, 10, 5, 5]  # [25 for li in range(layers*2)]  # [25, 25, 25]  # Polynomial orders.
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


