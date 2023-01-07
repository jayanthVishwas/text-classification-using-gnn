import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import re
import argparse
import datetime

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--cuda', type=bool, default=False, help='Use CUDA')
    args.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args.add_argument('--lr', type=float, default=0.02, help='learning rate.')
    args.add_argument('--model', type=str, default="GCN", choices=["GCN", "SAGE", "GAT"], help='model to use.')
    args.add_argument('--early_stopping', type=int, default=10, help='require early stopping.')
    args.add_argument('--dataset', type=str, default='mr', choices = ['20ng', 'R8', 'R52', 'ohsumed', 'mr'], help='dataset to train')

    arg_parser,_ = args.parse_known_args()
    return arg_parser

def preprocess_adj(adj):
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0])) 
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    degree_mat_inv_sqrt = sp.diags(d_inv)
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def construct_graph(adj):
    graph = DGLGraph()
    adj = preprocess_adj(adj)
    graph.add_nodes(adj.shape[0])
    graph.add_edges(adj.row, adj.col)
    adj_dense = adj.A
    adjd = np.ones((adj.shape[0]))
    for i in range(adj.shape[0]):
        adjd[i] = adjd[i] * np.sum(adj_dense[i, :])
    
    weight = torch.from_numpy(adj.data.astype(np.float32))
    graph.ndata['d'] = torch.from_numpy(adjd.astype(np.float32))
    graph.edata['w'] = weight

    return graph

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_corpus(dataset_dir):
    dir_prefix = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []

    for i in range(len(dir_prefix)):
        with open("{}/ind.{}.{}".format(dataset_dir, dir_prefix[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
    x, y, tx, ty, allx, ally, adj = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    train_idx_orig = parse_index_file("{}/ind.{}.train.index".format(dataset_dir))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def preprocess_features(features):

    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    return features.A


def print_log(msg='', end='\n'):
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
      + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)

    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]
        
    for line in lines:
        if line == lines[-1]:
            print('[' + t + '] ' + str(line), end=end)
        else: 
            print('[' + t + '] ' + str(line))



def loadWord2Vec(filename):

    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print_log('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
  
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()






