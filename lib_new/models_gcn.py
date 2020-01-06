from . import graph

import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import pandas as pd
import os, sys, time, collections, shutil,re
from pathlib import Path
from . import checkmat as checkmate

#NFEATURES = 28**2
#NCLASSES = 10


# Common methods for all models


class base_model(object):
    
    def __init__(self, config=None):
        self.regularizers = []
        self.sess = None
        if config is None:
            self.config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        else:
            self.config = config
    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size,) + data.shape[1:])
            tmp_data = data[begin:end,:,:]
            if type(tmp_data) is not np.ndarray:
                try:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                except:
                    print("Converting sparse matrix of nd array")

            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                ###edit by yuzhang
                ##print(batch_loss)
                if np.isnan(batch_loss): batch_loss = 0
                if np.isinf(batch_loss): batch_loss = 0
                #batch_loss[np.isnan(batch_loss)]=0
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
            
        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def evaluate(self, data, labels, sess=None, target_name=None, isTrain=False):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        sess = self._get_session(self.sess)
        if not isTrain:
            filename = tf.train.latest_checkpoint(os.path.join(self._get_path('checkpoints'), 'model'))
            self.op_saver.restore(sess, filename)
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        #print(predictions)

        if target_name is not None:
            try:
                print(sklearn.metrics.classification_report(labels, predictions, labels=range(len(target_name)), target_names=target_name))
                print('Confusion Matrix:')
                print(sklearn.metrics.confusion_matrix(labels, predictions, labels=range(len(target_name))))
            except:
                print("No corresponding assignment between true and predicted labels")

        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, accuracy, f1, loss

    def fit(self, train_data, train_labels, val_data, val_labels,best_checkpoint_dir=None):
        t_process, t_wall = time.process_time(), time.time()
        if self.sess is None:
            sess = tf.Session(config=self.config, graph=self.graph)
            self.sess = sess
        else:
            sess = self._get_session(self.sess)
        #shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        #writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'), exist_ok=True)
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)


        best_ckpt_saver = checkmate.BestCheckpointSaver(save_dir=path,num_to_keep=3,maximize=True,saver=self.op_saver)

        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        print('training with {} steps in total with batch_size={} and epochs={} for training_set={}:'.
              format(num_steps, self.batch_size, self.num_epochs,train_data.shape[0]))
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx,:,:], train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
            ##print("loss average:",loss_average)
            if np.isnan(loss_average): loss_average = 0
            if np.isinf(loss_average): loss_average = 0


            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess, isTrain=True)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))
                '''
                
                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)
                '''
                # Save model parameters (for evaluation).
                sys.stdout.flush()
                #self.op_saver.save(sess, path, global_step=self.global_step)
                best_ckpt_saver.handle(accuracy, sess, self.global_step)

        ##to include the last steps
        #best_ckpt_saver.handle(accuracy, sess, self.global_step)
        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        #writer.close()
        #sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    
    def build_graph(self, M_0,flag_input=True):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            if flag_input:
                # Inputs.
                with tf.name_scope('inputs'):
                    self.ph_data = tf.placeholder(tf.float32, (self.batch_size,) + M_0, 'data') ##(self.batch_size, M_0)
                    self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                    self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
            else:
                self.ph_data = None
                self.ph_dropout = 1

                # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate, self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        
        self.graph.finalize()
    
    def inference(self, data, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, dropout)
        return logits
    
    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, self.global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

            ################################
            ###choose different optimizer and learning rate
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                ##optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            try:
                grads = tf.gradients(loss, self.var_list)
                grads = list(zip(grads, self.var_list))
            except:
                grads = optimizer.compute_gradients(loss)
            ##grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=self.global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(config=self.config, graph=self.graph)
            ##filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            filename = tf.train.latest_checkpoint(os.path.join(self._get_path('checkpoints'), 'model'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_initial(self):
        if self.initial == 'normal':
            ##print("Using normal initializer for weights in graph-cnn!")
            initializer = tf.truncated_normal_initializer(0, 0.2) #0.1
        elif self.initial =='he':
            ###edited by yuzhang: using he initializer
            print("Using He initialization for weights in graph-cnn!")
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        return initializer

    def _weight_variable(self, shape, regularization=True):
        #initial = tf.truncated_normal_initializer(0, 0.2) #0.1
        initial = self._weight_initial()
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.2) ##0.1
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Graph convolutional

def bspline_basis(K, x, degree=3):
    """
    Return the B-spline basis.

    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis


class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
    
    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, config, L, F, K, p, M, filter='chebyshev5', brelu='b1relu', pool='mpool1', initial='normal', channel=1,
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                dir_name=''):
        super().__init__(config)
        
        # Verify the consistency w.r.t. the number of layers.
        try:
            assert len(L) >= len(F) == len(K) == len(p)
        except AssertionError:
            print(len(L))
            print(len(F), len(K), len(p))
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= np.sum(p_log2) #1 + # Enough coarsening levels for pool sizes.
        
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L
        
        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i-1] if i > 0 else M_0 ##if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
        
        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.initial = initial
        
        # Build the computational graph.
        self.build_graph((M_0,channel))  #M_0

    def filter_in_fourier(self, x, L, Fout, K, U, W):
        # TODO: N x F x M would avoid the permutations
        N, M, Fin = x.get_shape()
        N, M, Fin, Fout = int(N), int(M), int(Fin), int(Fout)
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # Transform to Fourier domain
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = tf.matmul(U, x)  # M x Fin*N
        x = tf.reshape(x, [M, Fin, N])  # M x Fin x N
        # Filter
        x = tf.matmul(W, x)  # for each feature
        x = tf.transpose(x)  # N x Fout x M
        x = tf.reshape(x, [N*Fout, M])  # N*Fout x M
        # Transform back to graph domain
        x = tf.matmul(x, U)  # N*Fout x M
        x = tf.reshape(x, [N, Fout, M])  # N x Fout x M
        return tf.transpose(x, perm=[0, 2, 1])  # N x M x Fout

    def fourier(self, x, L, Fout, K):
        ##assert K == L.shape[0]  # artificial but useful to compute number of parameters
        N, M, Fin = x.get_shape()
        N, M, Fin, Fout = int(N), int(M), int(Fin), int(Fout)
        # Fourier basis
        _, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)
        # Weights
        W = self._weight_variable([M, Fout, Fin], regularization=True)  ##changed by YU, use False
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    def spline(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin, Fout = int(N), int(M), int(Fin), int(Fout)
        # Fourier basis
        lamb, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)  # M x M
        # Spline basis
        B = bspline_basis(K, lamb, degree=3)  # M x K
        #B = bspline_basis(K, len(lamb), degree=3)  # M x K
        B = tf.constant(B, dtype=tf.float32)
        # Weights
        W = self._weight_variable([K, Fout*Fin], regularization=False)
        W = tf.matmul(B, W)  # M x Fout*Fin
        print(M, Fout, Fin)
        W = tf.reshape(W, [M, Fout, Fin])
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    def chebyshev2(self, x, L, Fout, K):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.
        
        Data: x of size N x M x F
            N: number of signals
            M: number of vertices
            F: number of features per signal per vertex
        """
        N, M, Fin = x.get_shape()
        N, M, Fin, Fout = int(N), int(M), int(Fin), int(Fout)
        # Rescale Laplacian. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        # Transform to Chebyshev basis
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        def chebyshev(x):
            return graph.chebyshev(L, x, K)
        x = tf.py_func(chebyshev, [x], [tf.float32])[0]  # K x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin, Fout = int(N), int(M), int(Fin), int(Fout)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=True)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, dropout):
        # Graph convolutional layers.
        ##x = tf.expand_dims(x, 2)  # N x M x F=1 (Fin=channel)
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])
        
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        ##x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        x = tf.reduce_mean(x, -1)
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)
        
        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return x


class finetuning_cgcnn(base_model):
    ####finetuning based on pretrained gcn models
    def __init__(self, config, checkpoint_dir, L, F, K, p, M, filter='chebyshev5', brelu='b1relu', pool='mpool1', initial=None, channel=1,
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name='',flag_tuning=False):
        super().__init__(config)
        # Verify the consistency w.r.t. the number of layers.
        try:
            assert len(L) >= len(F) == len(K) == len(p)
        except AssertionError:
            print(len(L))
            print(len(F), len(K), len(p))

        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        print('Fixed part: graph convolution layers')
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i + 1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                i, i + 1, L[i].shape[0], F[i], p[i], L[i].shape[0] * F[i] // p[i]))
            F_last = F[i - 1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i + 1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                    i + 1, L[i].shape[0], F[i], L[i].shape[0] * F[i]))

        print('Trainable part: fully connected layers')
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc - 1 else 'fc{}'.format(i + 1)
            print('  layer {}: {}'.format(Ngconv + i + 1, name))
            print('    representation: M_{} = {}'.format(Ngconv + i + 1, M[i]))
            M_last = M[i - 1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                Ngconv + i, Ngconv + i + 1, M_last, M[i], M_last * M[i]))
            print('    biases: M_{} = {}'.format(Ngconv + i + 1, M[i]))

        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.initial = initial
        self.fine_tuning = flag_tuning
        self.checkpoint_dir=checkpoint_dir

        # Build the computational graph.
        self.build_graph((M_0,channel),flag_input=False)  #M_0


    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, dropout):

        # Loading model from checkpint
        ##checkpoint_dir = self._get_path('checkpoints')
        ckp_path = self.checkpoint_dir + self.dir_name + '/model/'
        '''
        for model_file in sorted(Path(ckp_path).glob('model-*.meta')):
            model_name = os.path.basename(model_file).split('.')[0]
        '''
        lines = [line.rstrip('\n') for line in open(os.path.join(ckp_path, 'checkpoint'))]
        model_name = lines[0].replace('"', '').split(' ')[-1].split('/')[-1]
        if not os.path.isfile(ckp_path + model_name + ".meta"):
            model_name = lines[1].replace('"', '').split(' ')[-1].split('/')[-1]
        print(ckp_path + model_name + ".meta")

        ###prediction based on saved models
        sess = self.sess
        if sess is None:
            print('create new sessions')
            sess = tf.Session(config=self.config, graph=self.graph)

        # delete the current graph
        #tf.reset_default_graph()
        saver = tf.train.import_meta_graph(ckp_path + model_name + ".meta", clear_devices=True)
        saver.restore(sess, ckp_path + model_name)
        # saver.restore(sess, tf.train.latest_checkpoint(ckp_path))

        #ops = tf.get_default_graph().get_operations()
        ops = sess.graph.get_operations()
        # all the tensor informations
        tensors = [m.values() for m in ops]
        tensors_name = [m.name for m in ops]
        ##print(tensors)
        full_ops = np.unique([v.name.split('/')[0] for v in tf.trainable_variables()])
        train_layers = ['conv4','conv5','conv6']
        self.var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

        self.ph_data = sess.graph.get_tensor_by_name("inputs/data:0")
        self.ph_labels = sess.graph.get_tensor_by_name("inputs/labels:0")
        self.ph_dropout = sess.graph.get_tensor_by_name("inputs/dropout:0")

        conv = sess.graph.get_tensor_by_name("conv6/bias_relu/Relu:0")
        fc = sess.graph.get_tensor_by_name("fc1/Relu:0")

        if not self.fine_tuning:
            ##only if you want to fix previous conv layers
            conv = tf.stop_gradient(conv)  # It's an identity function
        '''
        ####how to include x in the model???
        feed_dict = {in_data: x}
        conv_out = sess.run([conv], feed_dict)
        '''
        # Fully connected hidden layers.
        conv_shape = conv.get_shape().as_list()
        x = tf.reshape(conv, [conv_shape[0], np.prod(conv_shape[1:])])
        C = self.M[-1]
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('newfc{}'.format(i + 1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('newlogits'):
            logits = self.fc(x, C, relu=False)

        for v in tf.trainable_variables():
            if re.search('new',v.name.split('/')[0]):
                self.var_list.append(v)
        print(self.var_list)

        return logits

    def _inference_test(self, x, dropout):

        # Loading model from checkpint
        ##checkpoint_dir = self._get_path('checkpoints')
        ckp_path = self.checkpoint_dir + self.dir_name + '/model/'
        '''
        for model_file in sorted(Path(ckp_path).glob('model-*.meta')):
            model_name = os.path.basename(model_file).split('.')[0]
        '''
        lines = [line.rstrip('\n') for line in open(os.path.join(ckp_path, 'checkpoint'))]
        model_name = lines[0].replace('"', '').split(' ')[-1].split('/')[-1]
        print(ckp_path + model_name + ".meta")

        ###prediction based on saved models
        sess = self.sess
        if sess is None:
            print('create new sessions')
            sess = tf.Session(config=self.config, graph=self.graph)

        # delete the current graph
        #tf.reset_default_graph()
        saver = tf.train.import_meta_graph(ckp_path + model_name + ".meta", clear_devices=True)
        saver.restore(sess, ckp_path + model_name)
        # saver.restore(sess, tf.train.latest_checkpoint(ckp_path))

        #ops = tf.get_default_graph().get_operations()
        ops = sess.graph.get_operations()
        # all the tensor informations
        tensors = [m.values() for m in ops]
        tensors_name = [m.name for m in ops]

        self.ph_data = sess.graph.get_tensor_by_name("inputs/data:0")
        self.ph_labels = sess.graph.get_tensor_by_name("inputs/labels:0")
        self.ph_dropout = sess.graph.get_tensor_by_name("inputs/dropout:0")

        conv = sess.graph.get_tensor_by_name("conv3/bias_relu/Relu:0")
        fc = sess.graph.get_tensor_by_name("fc1/Relu:0")

        if not self.fine_tuning:
            ##only if you want to fix previous conv layers
            conv = tf.stop_gradient(conv)  # It's an identity function

        for i in range(len(self.p)-3):
            with tf.variable_scope('newconv{}'.format(i+4)):
                with tf.name_scope('filter'):
                    conv = self.filter(conv, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    conv = self.brelu(conv)
                with tf.name_scope('pooling'):
                    conv = self.pool(conv, self.p[i])

        # Fully connected hidden layers.
        conv_shape = conv.get_shape().as_list()
        x = tf.reshape(conv, [conv_shape[0], np.prod(conv_shape[1:])])
        C = self.M[-1]
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('newfc{}'.format(i + 1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('newlogits'):
            logits = self.fc(x, C, relu=False)

        for v in tf.trainable_variables():
            if re.search('new',v.name.split('/')[0]):
                self.var_list.append(v)
        print(self.var_list)

        return logits

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, self.global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

            ################################
            ###choose different optimizer and learning rate
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                ##optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
                #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                #optimizer = tf.train.GradientDescentOptimizer(0.005)
                optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=0.001)

            grads = tf.gradients(loss, self.var_list)
            print('grads:', grads)
            grads = list(zip(grads, self.var_list))
            print('list:', grads)

            op_gradients = optimizer.apply_gradients(grads_and_vars=grads, global_step=self.global_step)

            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train


class model_perf(object):

    def __init__(s):
        s.names, s.params = set(), {}
        s.fit_accuracies, s.fit_losses, s.fit_time = {}, {}, {}
        s.train_accuracy, s.train_f1, s.train_loss = {}, {}, {}
        s.test_accuracy, s.test_f1, s.test_loss = {}, {}, {}

    def test(s, model, name, params, train_data, train_labels, val_data, val_labels, test_data, test_labels, target_name=None):
        s.params[name] = params
        s.fit_accuracies[name], s.fit_losses[name], s.fit_time[name] = \
                model.fit(train_data, train_labels, val_data, val_labels)
        string, s.train_accuracy[name], s.train_f1[name], s.train_loss[name] = \
                model.evaluate(train_data, train_labels, target_name=target_name)
        print('\ntrain {}\n'.format(string))

        string, s.test_accuracy[name], s.test_f1[name], s.test_loss[name] = \
                model.evaluate(test_data, test_labels, target_name=target_name)
        print('\ntest  {}\n'.format(string))
        sys.stdout.flush()
        s.names.add(name)

        return s

    def predict(s, ckp_path, test_data, test_labels, target_name=None, batch_size=128, trial_dura=17, flag_starttr=False,sub_name=None):

        ##ckp_path = Path(os.path.join(pathcheckpoints,modality,'win'+str(block_dura),method_str_new))
        ckp_path = str(ckp_path) + '/' + 'model/'
        '''
        for model_file in sorted(Path(ckp_path).glob('model-*.meta')):
            model_name = os.path.basename(model_file).split('.')[0]
        '''
        lines = [line.rstrip('\n') for line in open(os.path.join(ckp_path, 'checkpoint'))]
        model_name = lines[1].replace('"', '').split(' ')[-1].split('/')[-1]
        print(ckp_path + model_name + ".meta")

        pred_logits = []
        pred_labels = []
        pred_loss = 0
        data_size = test_data.shape[0]

        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(ckp_path + model_name + ".meta", clear_devices=True)
            saver.restore(sess, ckp_path + model_name)
            #saver.restore(sess, tf.train.latest_checkpoint(ckp_path))

            ops = sess.graph.get_operations()
            # all the tensor informations
            tensors = [m.values() for m in ops]
            tensors_name = [m.name for m in ops]

            in_data = sess.graph.get_tensor_by_name("inputs/data:0")
            in_label = sess.graph.get_tensor_by_name("inputs/labels:0")
            in_dropout = sess.graph.get_tensor_by_name("inputs/dropout:0")

            logits = sess.graph.get_tensor_by_name("logits/add:0")
            ##y_pred = tf.argmax(logits, axis=1)
            y_pred = sess.graph.get_tensor_by_name("prediction/ArgMax:0")
            loss = sess.graph.get_tensor_by_name("loss/add:0")

            # Now we are dupplicating the input in the first dimension
            for begin in range(0, data_size, batch_size):
                end = min([begin + batch_size, data_size])
                ##print(begin, end)

                batch_data = np.zeros((batch_size,) + test_data.shape[1:])
                tmp_data = test_data[begin:end, :, :]
                if type(tmp_data) is not np.ndarray:
                    try:
                        tmp_data = tmp_data.toarray()  # convert sparse matrices
                    except:
                        print("Converting sparse matrix of nd array")

                batch_data[:end - begin] = tmp_data
                batch_labels = np.zeros(batch_size)
                batch_labels[:end - begin] = test_labels[begin:end]
                ##print(batch_data.shape, tmp_data.shape, batch_labels.shape)

                feed_dict = {in_data: batch_data, in_label: batch_labels, in_dropout: 1}
                batch_logits, batch_pred, batch_loss = sess.run([logits, y_pred, loss], feed_dict)

                pred_logits.append(batch_logits)
                pred_labels.append(batch_pred)
                pred_loss += batch_loss

        pred_labels = np.stack(pred_labels,axis=0).flatten()[:len(test_labels)]
        pred_logits = np.stack(pred_logits, axis=0).flatten()[:len(test_labels)]

        print(sklearn.metrics.classification_report(test_labels, pred_labels, labels=range(len(target_name)), target_names=target_name))
        print('Confusion Matrix:')
        print(sklearn.metrics.confusion_matrix(test_labels, pred_labels, labels=range(len(target_name))))

        test_acc = []
        ncorrects = sum(pred_labels == test_labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(test_labels, pred_labels)
        f1 = 100 * sklearn.metrics.f1_score(test_labels, pred_labels, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(test_labels), f1, pred_loss)
        print(string)
        test_acc.append(accuracy)
        sys.stdout.flush()

        if sub_name is not None:
            print('\nGenerating subject-specific f1-score for task prediction...')
            print(pred_labels.shape, test_labels.shape, len(sub_name), len(sub_name)*6)
            try:
                y_pred = np.array(np.split(pred_labels, len(sub_name)))
                y_label = np.array(np.split(test_labels, len(sub_name)))
            except:
                sub_used = pred_labels.shape[0] // len(sub_name) * len(sub_name)
                y_pred = np.array(np.split(pred_labels[:sub_used,], len(sub_name)))
                y_label = np.array(np.split(test_labels[:sub_used,], len(sub_name)))

            test_acc = np.zeros((len(sub_name), len(target_name)+1))
            for subi in range(len(sub_name)):
                for li in range(len(target_name)):
                    trial_mask = y_label[subi, :] == li
                    f1 = sklearn.metrics.f1_score(y_label[subi, trial_mask], y_pred[subi, trial_mask], average='weighted')
                    test_acc[subi, li] = f1
                f1 = sklearn.metrics.f1_score(y_label[subi, :], y_pred[subi, :], average='weighted')
                test_acc[subi, -1] = f1  # f1
            result_df = pd.DataFrame()
            result_df['subject'] = sub_name
            for li,task in enumerate(target_name):
                result_df[task] = test_acc[:,li]
            result_df['avg'] = test_acc[:, -1]
            result_df.to_csv('train_logs/'+target_name[0].split('_')[-1]+'_f1score_testacc_'+str(len(sub_name))+'subjects.csv', sep='\t', encoding='utf-8', index=False)

        ####for each time point
        if flag_starttr:
            y_pred = np.reshape(pred_labels, (-1, trial_dura))
            y_label = np.reshape(test_labels, (-1, trial_dura))
            test_acc = np.zeros((len(target_name),trial_dura))
            for li in range(len(target_name)):
                print('\n',target_name[li],':')
                for ti in range(trial_dura):
                    trial_mask = y_label[:, ti] == li
                    ncorrects = sum(y_pred[trial_mask, ti] == y_label[trial_mask, ti])
                    accuracy = 100 * sklearn.metrics.accuracy_score(y_label[trial_mask, ti], y_pred[trial_mask, ti])
                    f1 = 100 * sklearn.metrics.f1_score(y_label[trial_mask, ti], y_pred[trial_mask, ti],average='weighted')
                    string = 'start_tr {:d} accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}'.format(ti, accuracy, ncorrects, np.sum(trial_mask), f1)
                    print(string)
                    test_acc[li,ti] = accuracy #f1

            print('\ntotal:')
            for ti in range(trial_dura):
                ncorrects = sum(y_pred[:, ti] == y_label[:, ti])
                accuracy = 100 * sklearn.metrics.accuracy_score(y_label[:, ti], y_pred[:, ti])
                f1 = 100 * sklearn.metrics.f1_score(y_label[:, ti], y_pred[:, ti], average='weighted')
                string = 'start_tr {:d} accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}'.format(ti, accuracy,ncorrects, len(y_pred), f1)
                print(string)
        return pred_logits, pred_labels, pred_loss, test_acc


    def predict_allmodel(s, ckp_path, test_data, test_labels, target_name=None, batch_size=128):

        ##ckp_path = Path(os.path.join(pathcheckpoints,modality,'win'+str(block_dura),method_str_new))

        for model_file in sorted(Path(ckp_path).glob('model-*.meta')):
            model_name = os.path.basename(model_file).split('.')[0]
            print(str(ckp_path) + '/' + model_name + ".meta")

            pred_logits = []
            pred_labels = []
            pred_loss = 0
            data_size = test_data.shape[0]

            tf.reset_default_graph()
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(str(ckp_path) + '/' + model_name + ".meta", clear_devices=True)
                saver.restore(sess, str(ckp_path) + '/' + model_name)
                #saver.restore(sess, tf.train.latest_checkpoint(ckp_path))

                ops = sess.graph.get_operations()
                # all the tensor informations
                tensors = [m.values() for m in ops]
                tensors_name = [m.name for m in ops]

                in_data = sess.graph.get_tensor_by_name("inputs/data:0")
                in_label = sess.graph.get_tensor_by_name("inputs/labels:0")
                in_dropout = sess.graph.get_tensor_by_name("inputs/dropout:0")

                logits = sess.graph.get_tensor_by_name("logits/add:0")
                ##y_pred = tf.argmax(logits, axis=1)
                y_pred = sess.graph.get_tensor_by_name("prediction/ArgMax:0")
                loss = sess.graph.get_tensor_by_name("loss/add:0")

                # Now we are dupplicating the input in the first dimension
                for begin in range(0, data_size, batch_size):
                    end = min([begin + batch_size, data_size])
                    ##print(begin, end)

                    batch_data = np.zeros((batch_size,) + test_data.shape[1:])
                    tmp_data = test_data[begin:end, :, :]
                    if type(tmp_data) is not np.ndarray:
                        try:
                            tmp_data = tmp_data.toarray()  # convert sparse matrices
                        except:
                            print("Converting sparse matrix of nd array")

                    batch_data[:end - begin] = tmp_data
                    batch_labels = np.zeros(batch_size)
                    batch_labels[:end - begin] = test_labels[begin:end]
                    ##print(batch_data.shape, tmp_data.shape, batch_labels.shape)

                    feed_dict = {in_data: batch_data, in_label: batch_labels, in_dropout: 1}
                    batch_logits, batch_pred, batch_loss = sess.run([logits, y_pred, loss], feed_dict)

                    pred_logits.append(batch_logits)
                    pred_labels.append(batch_pred)
                    pred_loss += batch_loss

            pred_labels = np.stack(pred_labels,axis=0).flatten()[:len(test_labels)]
            pred_logits = np.stack(pred_logits, axis=0).flatten()[:len(test_labels)]

            print(sklearn.metrics.classification_report(test_labels, pred_labels, labels=range(len(target_name)), target_names=target_name))
            print('Confusion Matrix:')
            print(sklearn.metrics.confusion_matrix(test_labels, pred_labels, labels=range(len(target_name))))

            ncorrects = sum(pred_labels == test_labels)
            accuracy = 100 * sklearn.metrics.accuracy_score(test_labels, pred_labels)
            f1 = 100 * sklearn.metrics.f1_score(test_labels, pred_labels, average='weighted')
            string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                    accuracy, ncorrects, len(test_labels), f1, pred_loss)
            print(string)
            sys.stdout.flush()
        return pred_logits, pred_labels, pred_loss

    def show(s, fontsize=None):
        import matplotlib.pyplot as plt

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
