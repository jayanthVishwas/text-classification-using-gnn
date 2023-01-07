from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn
from dgl import DGLGraph
import numpy as np

from src.utils.utils import *
from src.models.gcn import GCN
from src.models.gat import GAT
from src.trainer.trainer import Trainer

import argparse

if __name__ == '__main__':
    args = get_args()
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    adj_m, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus('mr')

    features = sp.identity(features.shape[0])
    features = preprocess_features(features)

    adj_dense = torch.from_numpy

    graph = construct_graph(adj_m)

    if args.cuda:
        graph = graph.to(torch.device('cuda:0'))
    

    torch_features = torch.from_numpy(features.astype(np.float32))
    torch_y_train = torch.from_numpy(y_train)
    torch_y_val = torch.from_numpy(y_val)
    torch_y_test = torch.from_numpy(y_test)
    torch_train_mask = torch.from_numpy(train_mask.astype(np.float32))
    torch_train_mask_t = torch.transpose(torch.unsqueeze(torch_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
    support = [preprocess_adj(adj_m)]
    n_supports = 1
    torch_support = []
    for i in range(len(support)):
        torch_support.append(torch.Tensor(support[i]))
    
    if args.model == 'GCN':
        model = GCN(input_dim = features.shape[0], hidden_dim = 128, n_classes = y_train.shape[1])
    elif args.model == 'GAT':
        model = GAT(input_dim = features.shape[0], hidden_dim = 128, n_classes = y_train.shape[1])

    if args.cuda and torch.cuda.is_available():
        torch_features = torch_features.cuda()
        torch_y_train = torch_y_train.cuda()
        torch_train_mask = torch_train_mask.cuda()
        torch_train_mask_t = torch_train_mask_t.cuda()
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(args, model, optimizer, criterion, features, y_train, y_val, y_test, train_mask, val_mask )

    trainer.train()

    test_loss, test_acc, pred, labels = trainer.eval(torch_features, torch_y_test, test_mask)

    print("Test set results: loss= {:.4f} accuracy= {:.4f}".format(test_loss, test_acc))

    test_pred = []
    test_labels = []

    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(np.argmax(labels[i]))
    
    print("Test set results: accuracy= {:.4f}".format(metrics.accuracy_score(test_labels, test_pred)))
    print("Test set results: precision= {:.4f}".format(metrics.precision_score(test_labels, test_pred, average='macro')))
    print("Test set results: recall= {:.4f}".format(metrics.recall_score(test_labels, test_pred, average='macro')))
    print("Test set results: f1= {:.4f}".format(metrics.f1_score(test_labels, test_pred, average='macro')))







