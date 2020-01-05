import gensim
import sklearn, sklearn.datasets
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time, re, sys
import pandas as pd

# Helpers to process text documents.


class TextDataset(object):
    def clean_text(self, num='substitute'):
        # TODO: stemming, lemmatisation
        for i,doc in enumerate(self.documents):
            # Digits.
            if num is 'spell':
                doc = doc.replace('0', ' zero ')
                doc = doc.replace('1', ' one ')
                doc = doc.replace('2', ' two ')
                doc = doc.replace('3', ' three ')
                doc = doc.replace('4', ' four ')
                doc = doc.replace('5', ' five ')
                doc = doc.replace('6', ' six ')
                doc = doc.replace('7', ' seven ')
                doc = doc.replace('8', ' eight ')
                doc = doc.replace('9', ' nine ')
            elif num is 'substitute':
                # All numbers are equal. Useful for embedding (countable words) ?
                doc = re.sub('(\\d+)', ' NUM ', doc)
            elif num is 'remove':
                # Numbers are uninformative (they are all over the place). Useful for bag-of-words ?
                # But maybe some kind of documents contain more numbers, e.g. finance.
                # Some documents are indeed full of numbers. At least in 20NEWS.
                doc = re.sub('[0-9]', ' ', doc)
            # Remove everything except a-z characters and single space.
            doc = doc.replace('$', ' dollar ')
            doc = doc.lower()
            doc = re.sub('[^a-z]', ' ', doc)
            doc = ' '.join(doc.split())  # same as doc = re.sub('\s{2,}', ' ', doc)
            self.documents[i] = doc

    def vectorize(self, **params):
        # TODO: count or tf-idf. Or in normalize ?
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(**params)
        self.data = vectorizer.fit_transform(self.documents)
        self.vocab = vectorizer.get_feature_names()
        assert len(self.vocab) == self.data.shape[1]
    
    def data_info(self, show_classes=False):
        N, M = self.data.shape
        sparsity = self.data.nnz / N / M * 100
        print('N = {} documents, M = {} words, sparsity={:.4f}%'.format(N, M, sparsity))
        if show_classes:
            for i in range(len(self.class_names)):
                num = sum(self.labels == i)
                print('  {:5d} documents in class {:2d} ({})'.format(num, i, self.class_names[i]))
        
    def show_document(self, i):
        label = self.labels[i]
        name = self.class_names[label]
        try:
            text = self.documents[i]
            wc = len(text.split())
        except AttributeError:
            text = None
            wc = 'N/A'
        print('document {}: label {} --> {}, {} words'.format(i, label, name, wc))
        try:
            vector = self.data[i,:]
            for j in range(vector.shape[1]):
                if vector[0,j] != 0:
                    print('  {:.2f} "{}" ({})'.format(vector[0,j], self.vocab[j], j))
        except AttributeError:
            pass
        return text
    
    def keep_documents(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.documents = [self.documents[i] for i in idx]
        self.labels = self.labels[idx]
        self.data = self.data[idx,:]

    def keep_words(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.data = self.data[:,idx]
        self.vocab = [self.vocab[i] for i in idx]
        try:
            self.embeddings = self.embeddings[idx,:]
        except AttributeError:
            pass

    def remove_short_documents(self, nwords, vocab='selected'):
        """Remove a document if it contains less than nwords."""
        if vocab is 'selected':
            # Word count with selected vocabulary.
            wc = self.data.sum(axis=1)
            wc = np.squeeze(np.asarray(wc))
        elif vocab is 'full':
            # Word count with full vocabulary.
            wc = np.empty(len(self.documents), dtype=np.int)
            for i,doc in enumerate(self.documents):
                wc[i] = len(doc.split())
        idx = np.argwhere(wc >= nwords).squeeze()
        self.keep_documents(idx)
        return wc
        
    def keep_top_words(self, M, Mprint=20):
        """Keep in the vocaluary the M words who appear most often."""
        freq = self.data.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:M]
        self.keep_words(idx)
        print('most frequent words')
        for i in range(Mprint):
            print('  {:3d}: {:10s} {:6d} counts'.format(i, self.vocab[i], freq[idx][i]))
        return freq[idx]
    
    def normalize(self, norm='l1'):
        """Normalize data to unit length."""
        # TODO: TF-IDF.
        data = self.data.astype(np.float64)
        self.data = sklearn.preprocessing.normalize(data, axis=1, norm=norm)
        
    def embed(self, filename=None, size=100):
        """Embed the vocabulary using pre-trained vectors."""
        if filename:
            model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True)
            size = model.vector_size
        else:
            class Sentences(object):
                def __init__(self, documents):
                    self.documents = documents
                def __iter__(self):
                    for document in self.documents:
                        yield document.split()
            model = gensim.models.Word2Vec(Sentences(self.documents), size)
        self.embeddings = np.empty((len(self.vocab), size))
        keep = []
        not_found = 0
        for i,word in enumerate(self.vocab):
            try:
                self.embeddings[i,:] = model[word]
                keep.append(i)
            except KeyError:
                not_found += 1
        print('{} words not found in corpus'.format(not_found, i))
        self.keep_words(keep)

class Text20News(TextDataset):
    def __init__(self, **params):
        dataset = sklearn.datasets.fetch_20newsgroups(**params)
        self.documents = dataset.data
        self.labels = dataset.target
        self.class_names = dataset.target_names
        assert max(self.labels) + 1 == len(self.class_names)
        N, C = len(self.documents), len(self.class_names)
        print('N = {} documents, C = {} classes'.format(N, C))

class TextRCV1(TextDataset):
    def __init__(self, **params):
        dataset = sklearn.datasets.fetch_rcv1(**params)
        self.data = dataset.data
        self.target = dataset.target
        self.class_names = dataset.target_names
        assert len(self.class_names) == 103  # 103 categories according to LYRL2004
        N, C = self.target.shape
        assert C == len(self.class_names)
        print('N = {} documents, C = {} classes'.format(N, C))

    def remove_classes(self, keep):
        ## Construct a lookup table for labels.
        labels_row = []
        labels_col = []
        class_lookup = {}
        for i,name in enumerate(self.class_names):
            class_lookup[name] = i
        self.class_names = keep

        # Index of classes to keep.
        idx_keep = np.empty(len(keep))
        for i,cat in enumerate(keep):
            idx_keep[i] = class_lookup[cat]
        self.target = self.target[:,idx_keep]
        assert self.target.shape[1] == len(keep)

    def show_doc_per_class(self, print_=False):
        """Number of documents per class."""
        docs_per_class = np.array(self.target.astype(np.uint64).sum(axis=0)).squeeze()
        print('categories ({} assignments in total)'.format(docs_per_class.sum()))
        if print_:
            for i,cat in enumerate(self.class_names):
                print('  {:5s}: {:6d} documents'.format(cat, docs_per_class[i]))
        plt.figure(figsize=(17,5))
        plt.plot(sorted(docs_per_class[::-1]),'.')

    def show_classes_per_doc(self):
        """Number of classes per document."""
        classes_per_doc = np.array(self.target.sum(axis=1)).squeeze()
        plt.figure(figsize=(17,5))
        plt.plot(sorted(classes_per_doc[::-1]),'.')

    def select_documents(self):
        classes_per_doc = np.array(self.target.sum(axis=1)).squeeze()
        self.target = self.target[classes_per_doc==1]
        self.data = self.data[classes_per_doc==1, :]

        # Convert labels from indicator form to single value.
        N, C = self.target.shape
        target = self.target.tocoo()
        self.labels = target.col
        assert self.labels.min() == 0
        assert self.labels.max() == C - 1

        # Bruna and Dropout used 2 * 201369 = 402738 documents. Probably the difference btw v1 and v2.
        #return classes_per_doc

### Helpers to quantify classifier's quality.


def baseline(train_data, train_labels, test_data, test_labels, omit=[]):
    """Train various classifiers to get a baseline."""
    clf, train_accuracy, test_accuracy, train_f1, test_f1, exec_time = [], [], [], [], [], []
    clf.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=10))
    clf.append(sklearn.linear_model.LogisticRegression())
    clf.append(sklearn.naive_bayes.BernoulliNB(alpha=.01))
    clf.append(sklearn.ensemble.RandomForestClassifier())
    clf.append(sklearn.naive_bayes.MultinomialNB(alpha=.01))
    clf.append(sklearn.linear_model.RidgeClassifier())
    clf.append(sklearn.svm.LinearSVC())
    for i,c in enumerate(clf):
        if i not in omit:
            t_start = time.process_time()
            c.fit(train_data, train_labels)
            train_pred = c.predict(train_data)
            test_pred = c.predict(test_data)
            train_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(train_labels, train_pred)))
            test_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(test_labels, test_pred)))
            train_f1.append('{:5.2f}'.format(100*sklearn.metrics.f1_score(train_labels, train_pred, average='weighted')))
            test_f1.append('{:5.2f}'.format(100*sklearn.metrics.f1_score(test_labels, test_pred, average='weighted')))
            exec_time.append('{:5.2f}'.format(time.process_time() - t_start))
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
    print('Execution time:      {}'.format(' '.join(exec_time)))

def grid_search(params, grid_params, train_data, train_labels, val_data,
        val_labels, test_data, test_labels, model):
    """Explore the hyper-parameter space with an exhaustive grid search."""
    params = params.copy()
    train_accuracy, test_accuracy, train_f1, test_f1 = [], [], [], []
    grid = sklearn.model_selection.ParameterGrid(grid_params)
    print('grid search: {} combinations to evaluate'.format(len(grid)))
    for grid_params in grid:
        params.update(grid_params)
        name = '{}'.format(grid)
        print('\n\n  {}  \n\n'.format(grid_params))
        m = model(params)
        m.fit(train_data, train_labels, val_data, val_labels)
        string, accuracy, f1, loss = m.evaluate(train_data, train_labels)
        train_accuracy.append('{:5.2f}'.format(accuracy)); train_f1.append('{:5.2f}'.format(f1))
        print('train {}'.format(string))
        string, accuracy, f1, loss = m.evaluate(test_data, test_labels)
        test_accuracy.append('{:5.2f}'.format(accuracy)); test_f1.append('{:5.2f}'.format(f1))
        print('test  {}'.format(string))
    print('\n\n')
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
    for i,grid_params in enumerate(grid):
        print('{} --> {} {} {} {}'.format(grid_params, train_accuracy[i], test_accuracy[i], train_f1[i], test_f1[i]))


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
        import tensorflow as tf
        import os
        from pathlib import Path
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
        import sklearn
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
            result_df.to_csv('train_log/'+target_name[0].split('_')[-1]+'_f1score_testacc_'+str(len(sub_name))+'subjects.csv', sep='\t', encoding='utf-8', index=False)

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
        import tensorflow as tf
        import os
        from pathlib import Path
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
            import sklearn
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
