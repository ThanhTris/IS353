import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from collections import defaultdict


"""
    Utility functions to handle data and evaluate model (YelpChi only).
"""


def load_data():
    """
    Load YelpChi graph, node features, and labels.

    YelpChi Dataset:
      - Nodes: 45,954 reviews (each node = one Yelp review)
      - Labels: 0 = Legit, 1 = Spam  (~14.5% spam)
      - Features: 32-dim behavioral features per review node
      - Relations:
          homo        — homogeneous graph (all edges merged)
          yelp_rur    — Review-User-Review (same user wrote both reviews)
          yelp_rtr    — Review-Text-Review (similar text content)
          yelp_rsr    — Review-Star-Review (same star rating given)

    :returns: [homo_adj, rur_adj, rtr_adj, rsr_adj], feat_data, labels
    """
    prefix = 'data/'
    data_file = loadmat(prefix + 'YelpChi.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as f:
        homo = pickle.load(f)
    with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as f:
        relation1 = pickle.load(f)
    with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as f:
        relation2 = pickle.load(f)
    with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as f:
        relation3 = pickle.load(f)

    return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
    """
    Row-normalize sparse matrix.
    Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list and save as pickle.
    Adds a self-loop to each node.
    :param sp_matrix: the sparse matrix
    :param filename: output pickle filename
    """
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)


def pos_neg_split(nodes, labels):
    """
    Split nodes into positive (spam) and negative (legit) sets.
    :param nodes: list of node indices
    :param labels: corresponding label array
    :returns: (pos_nodes, neg_nodes)
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])
    return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
    """
    Under-sample negative (legit) nodes to balance training batches.
    :param pos_nodes: list of positive (spam) node indices
    :param neg_nodes: list of negative (legit) node indices
    :param scale: ratio of neg to pos to keep
    :return: combined list of sampled batch nodes
    """
    aux_nodes = cp.deepcopy(neg_nodes)
    aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes) * scale))
    return pos_nodes + aux_nodes


def test_sage(test_cases, labels, model, batch_size):
    """
    Evaluate GraphSAGE baseline performance.
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    gnn_list = []

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes)
        f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
        acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
        recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
        gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_list))
    print(f"  GNN F1      : {f1_gnn / test_batch_num:.4f}")
    print(f"  GNN Accuracy: {acc_gnn / test_batch_num:.4f}")
    print(f"  GNN Recall  : {recall_gnn / test_batch_num:.4f}")
    print(f"  GNN AUC     : {auc_gnn:.4f}")
    print(f"  GNN AP      : {ap_gnn:.4f}")


def test_care(test_cases, labels, model, batch_size):
    """
    Evaluate CARE-GNN and its Simi module.
    :returns: (auc_gnn, auc_label, recall_gnn, recall_label,
               gnn_score_array, label_score_array)
              The last two are probability scores for the positive class —
              used for Precision-Recall curve and t-SNE visualization.
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    f1_label1 = 0.0
    acc_label1 = 0.0
    recall_label1 = 0.0
    gnn_list = []
    label_list1 = []

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)

        f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
        acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
        recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
        f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
        acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
        recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

        gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
        label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_list))
    auc_label1 = roc_auc_score(labels, np.array(label_list1))
    ap_label1 = average_precision_score(labels, np.array(label_list1))

    print(f"  [GNN]   F1={f1_gnn/test_batch_num:.4f}  Acc={acc_gnn/test_batch_num:.4f}"
          f"  Recall={recall_gnn/test_batch_num:.4f}  AUC={auc_gnn:.4f}  AP={ap_gnn:.4f}")
    print(f"  [Label] F1={f1_label1/test_batch_num:.4f}  Acc={acc_label1/test_batch_num:.4f}"
          f"  Recall={recall_label1/test_batch_num:.4f}  AUC={auc_label1:.4f}  AP={ap_label1:.4f}")

    return (auc_gnn, auc_label1,
            recall_gnn / test_batch_num,   # averaged recall (fix: không tích lũy)
            recall_label1 / test_batch_num,
            np.array(gnn_list), np.array(label_list1))