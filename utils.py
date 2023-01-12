import sys
import os
import torch
import random
import math

import csv
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score, recall_score, \
    precision_score

from torchmetrics import RetrievalHitRate

import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
from numpy.random import default_rng
from scipy.sparse import lil_matrix
from scipy import sparse
import dgl
import torch.distributions as tdist
from scipy.stats import multivariate_normal


def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds + '_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds + '_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()


def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch + 1, epochs, index,
                                                                                            batches, loss.item(),
                                                                                            len(visited_nodes),
                                                                                            len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()

        max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification, max_vali_f1


def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.SGD(params, lr=0.7)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings

        embs_batch = graphSage(nodes_batch)

        if learn_method == 'sup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                         len(visited_nodes), len(train_nodes)))

        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

    return graphSage, classification


###############################################


class node_mlp(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """

    def __init__(self, input, layers=[16, 16], normalize=False, dropout_rate=0):
        """
        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super().__init__()
        # super(node_mlp, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input, layers[0])])

        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i + 1]))

        self.norm_layers = None
        if normalize:
            self.norm_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(c) for c in [input] + layers])
        self.dropout = torch.nn.Dropout(dropout_rate)
        # self.reset_parameters()

    def forward(self, in_tensor, activation=torch.tanh):
        h = in_tensor
        for i in range(len(self.layers)):
            if self.norm_layers != None:
                if len(h.shape) == 2:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h = h.reshape(-1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h = h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            h = activation(h)
        return h


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges_old(adj, feature):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    index = list(range(train_edges.shape[0]))
    np.random.shuffle(index)
    train_edges_true = train_edges[index[0:num_val]]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])
    # print(test_edges_false)
    # print(val_edges_false)
    # print(test_edges)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    ignore_edges_inx = [list(np.array(val_edges_false)[:, 0]), list(np.array(val_edges_false)[:, 1])]
    ignore_edges_inx[0].extend(val_edges[:, 0])
    ignore_edges_inx[1].extend(val_edges[:, 1])
    import copy

    val_edge_idx = copy.deepcopy(ignore_edges_inx)
    ignore_edges_inx[0].extend(test_edges[:, 0])
    ignore_edges_inx[1].extend(test_edges[:, 1])
    ignore_edges_inx[0].extend(np.array(test_edges_false)[:, 0])
    ignore_edges_inx[1].extend(np.array(test_edges_false)[:, 1])

    np.save('train.npy', train_edges)
    np.save('valid.npy', val_edges)
    np.save('test.npy', test_edges)

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, list(
        train_edges_true), train_edges_false, ignore_edges_inx, val_edge_idx


# objective Function
def optimizer_VAE(pred, labels, std_z, mean_z, num_nodes, pos_wight, norm, reconstructed_feat, feat_train,
                  inductive_task):
    if inductive_task == 'node_classification_us':
        lambda_a = 1
        lambda_x = 0

    val_poterior_cost = 0
    posterior_cost = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight)

    posterior_cost_feat = norm * F.binary_cross_entropy_with_logits(feat_train, reconstructed_feat,
                                                                    pos_weight=pos_wight)

    z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))
    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])

    return z_kl, lambda_a * posterior_cost, acc, val_poterior_cost, lambda_x * posterior_cost_feat



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


import copy


def make_test_train_gpu(adj, feat, split=[]):
    if len(split) == 0:
        num_test = int(np.floor(feat.shape[0] / 3.))
        num_val = int(np.floor(feat.shape[0] / 6.))
        rng = default_rng()
        numbers = rng.choice(feat.shape[0], feat.shape[0], replace=False)
        test_nodes = numbers[:num_test]
        val_nodes = numbers[-num_val:]
        train_nodes = numbers[num_val + num_test:]
    else:
        train_nodes = split[0]
        val_nodes = split[1]
        test_nodes = split[2]

    # create feature matrix
    feat_test = np.zeros((len(test_nodes), feat.shape[1]))
    feat_val = np.zeros((len(train_nodes), feat.shape[1]))

    feat_np = feat.cpu().data.numpy()
    feat_train = feat_np[train_nodes, :]
    feat_val = feat_np[val_nodes, :]
    feat_test = feat_np[test_nodes, :]

    # create adj
    adj_train = np.zeros((len(train_nodes), len(train_nodes)))
    adj_test = np.zeros((len(test_nodes), len(test_nodes)))
    adj_val = np.zeros((len(train_nodes), len(train_nodes)))

    adj_train = adj[train_nodes, :][:, train_nodes]
    adj_val = adj[val_nodes, :][:, val_nodes]
    adj_test = adj[test_nodes, :][:, test_nodes]

    train_true_i = adj_train.nonzero()[0]
    train_true_j = adj_train.nonzero()[1]
    train_true = [[train_true_i[x], train_true_j[x]] for x in range(len(train_true_i))]

    val_true_i = adj_val.nonzero()[0]
    val_true_j = adj_val.nonzero()[1]
    val_true = [[val_true_i[x], val_true_j[x]] for x in range(len(val_true_i))]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    val_edges_false = []
    while len(val_edges_false) < len(val_true):
        idx_i = np.random.randint(0, adj_val.shape[0])
        idx_j = np.random.randint(0, adj_val.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], np.array(val_true)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(train_true):
        idx_i = np.random.randint(0, adj_train.shape[0])
        idx_j = np.random.randint(0, adj_train.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], np.array(train_true)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    assert (len(val_edges_false) == len(val_true))

    return adj_train, adj_val, adj_test, feat_train, feat_val, feat_test, train_true, train_edges_false, val_true, val_edges_false


####################### NIPS UTILS ##########################


def test_(number_of_samples, model, graph_size, max_size, path_to_save_g, device, remove_self=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, max_size, model.latent_dim]))
            z = torch.randn_like(z)
            start_time = time.time()
            if type(model.decode) == GRAPHITdecoder:
                pass
                # adj_logit = model.decode(z.float(), features)
            elif type(model.decode) == RNNDecoder:
                adj_logit = model.decode(z.to(device).float(), [g_size])
            elif type(model.decode) in (FCdecoder, FC_InnerDOTdecoder):
                g_size = max_size
                z = torch.tensor(numpy.random.normal(size=[1, max_size, model.latent_dim]))
                z = torch.randn_like(z)
                adj_logit = model.decode(z.to(device).float())
            else:
                adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            sample_graph = sample_graph[:g_size, :g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_matrix(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g + str(g_size) + str(j) + dataset
            # plot and save the generated graph
            plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            plotter.plotG(G, "generated" + dataset, file_name=f_name + "_ConnectedComponnents")
    return generated_graph_list

    # save to pickle file


def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, num_nodes,
                 alpha, reconstructed_adj_logit, pos_wight, norm, node_num, reconstructed_feat, feat_train,
                 inductive_task, ignore_indexes=None):
    if inductive_task == 'node_classification_us':
        lambda_a = 0
        lambda_x = 200

    if ignore_indexes == None:
        loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(),
                                                                           targert_adj.float(), pos_weight=pos_wight)
    else:
        loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(),
                                                                           targert_adj.float(), pos_weight=pos_wight,
                                                                           reduction='none')
        loss[0][ignore_indexes[1], ignore_indexes[0]] = 0
        loss = loss.mean()
    norm = mean.shape[0] * mean.shape[1] * mean.shape[2]
    kl = (1 / norm) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum() / float(
        reconstructed_adj.shape[0] * reconstructed_adj.shape[1] * reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []
    for i in range(len(target_kernel_val)):
        l = torch.nn.MSELoss()
        step_loss = l(reconstructed_kernel_val[i].float(), target_kernel_val[i].float())
        each_kernel_loss.append(step_loss.cpu().detach().numpy() * alpha[i])
        kernel_diff += l(reconstructed_kernel_val[i].float(), target_kernel_val[i].float()) * alpha[i]
    each_kernel_loss.append(loss.cpu().detach().numpy() * alpha[-2])
    each_kernel_loss.append(kl.cpu().detach().numpy() * alpha[-1])
    kernel_diff += loss * alpha[-2]
    kernel_diff += kl * alpha[-1]

    device = reconstructed_adj_logit.get_device()
    feat_loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_feat.to(device),
                                                                            torch.from_numpy(feat_train).to(device),
                                                                            pos_weight=pos_wight)
    return kl, loss, acc, kernel_diff * lambda_a, each_kernel_loss, feat_loss * lambda_x


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


class Datasets():
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_adjs, self_for_none, list_Xs, padding=True, Max_num=None):
        """
        :param list_adjs: a list of adjacency in sparse format
        :param list_Xs: a list of node feature matrix
        """
        'Initialization'
        self.paading = padding
        self.list_Xs = list_Xs
        self.list_adjs = list_adjs
        self.toatl_num_of_edges = 0
        self.max_num_nodes = 0
        for i, adj in enumerate(list_adjs):
            list_adjs[i] = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            list_adjs[i] += sp.eye(list_adjs[i].shape[0])
            if self.max_num_nodes < adj.shape[0]:
                self.max_num_nodes = adj.shape[0]
            self.toatl_num_of_edges += adj.sum().item()
            # if list_Xs!=None:
            #     self.list_adjs[i], list_Xs[i] = self.permute(list_adjs[i], list_Xs[i])
            # else:
            #     self.list_adjs[i], _ = self.permute(list_adjs[i], None)
        if Max_num != None:
            self.max_num_nodes = Max_num
        self.processed_Xs = []
        self.processed_adjs = []
        self.num_of_edges = []
        for i in range(self.__len__()):
            a, x, n = self.process(i, self_for_none)
            self.processed_Xs.append(x)
            self.processed_adjs.append(a)
            self.num_of_edges.append(n)
        if list_Xs != None:
            self.feature_size = list_Xs[0].shape[1]
        else:
            self.feature_size = self.max_num_nodes

    def get(self, shuffle=True):
        indexces = list(range(self.__len__()))
        random.shuffle()
        return [self.processed_adjs[i] for i in indexces], [self.processed_Xs[i] for i in indexces]

    def get__(self, from_, to_, self_for_none):
        adj_s = []
        x_s = []
        num_nodes = []
        # padded_to = max([self.list_adjs[i].shape[1] for i in range(from_, to_)])
        # padded_to = 225
        for i in range(from_, to_):
            adj, x, num_node = self.process(i, self_for_none)  # , padded_to)
            adj_s.append(adj)
            x_s.append(x)
            num_nodes.append(num_node)
        return adj_s, x_s, num_nodes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_adjs)

    def process(self, index, self_for_none, padded_to=None, ):

        num_nodes = self.list_adjs[index].shape[0]
        if self.paading == True:
            max_num_nodes = self.max_num_nodes if padded_to == None else padded_to
        else:
            max_num_nodes = num_nodes
        adj_padded = lil_matrix((max_num_nodes, max_num_nodes))  # make the size equal to maximum graph
        adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index][:, :]
        adj_padded -= sp.dia_matrix((adj_padded.diagonal()[np.newaxis, :], [0]), shape=adj_padded.shape)
        if self_for_none:
            adj_padded += sp.eye(max_num_nodes)
        else:
            adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
        # adj_padded+= sp.eye(max_num_nodes)

        if self.list_Xs == None:
            # if the feature is not exist we use identical matrix
            X = np.identity(max_num_nodes)

        else:
            # ToDo: deal with data with diffrent number of nodes
            X = self.list_Xs[index].toarray()

        # adj_padded, X = self.permute(adj_padded, X)

        # Converting sparse matrix to sparse tensor
        coo = adj_padded.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        adj_padded = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        X = torch.tensor(X, dtype=torch.float32)

        return adj_padded.reshape(1, *adj_padded.shape), X.reshape(1, *X.shape), num_nodes

    def permute(self, list_adj, X):
        p = list(range(list_adj.shape[0]))
        np.random.shuffle(p)
        # for i in range(list_adj.shape[0]):
        #     list_adj[:, i] = list_adj[p, i]
        #     X[:, i] = X[p, i]
        # for i in range(list_adj.shape[0]):
        #     list_adj[i, :] = list_adj[i, p]
        #     X[i, :] = X[i, p]
        list_adj[:, :] = list_adj[p, :]
        list_adj[:, :] = list_adj[:, p]
        if X != None:
            X[:, :] = X[p, :]
            X[:, :] = X[:, p]
        return list_adj, X

    def shuffle(self):
        indx = list(range(len(self.list_adjs)))
        np.random.shuffle(indx)
        if self.list_Xs != None:
            self.list_Xs = [self.list_Xs[i] for i in indx]
        self.list_adjs = [self.list_adjs[i] for i in indx]

    def __getitem__(self, index):
        'Generates one sample of data'
        # return self.processed_adjs[index], self.processed_Xs[index],torch.tensor(self.list_adjs[index].todense(), dtype=torch.float32)
        return self.processed_adjs[index], self.processed_Xs[index]


# objective Function
def optimizer_VAE_pn(pred, labels, std_z, mean_z, num_nodes, pos_wight, norm):
    val_poterior_cost = 0
    posterior_cost = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight)

    z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))

    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])
    return z_kl, posterior_cost, acc, val_poterior_cost


def roc_auc_estimator(target_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    counter = 0
    for edge in target_edges:
        prediction.append(reconstructed_adj[edge[0], edge[1]])
        prediction.append(reconstructed_adj[edge[1], edge[0]])
        true_label.append(origianl_agjacency[edge[0], edge[1]])
        true_label.append(origianl_agjacency[edge[1], edge[0]])

    pred = np.array(prediction)
    pred[pred > .5] = 1.0
    pred[pred < .5] = 0.0
    pred = pred.astype(int)


    precision = precision_score(y_pred=pred, y_true=true_label)
    recall = recall_score(y_pred=pred, y_true=true_label)
    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)

    hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:]
    HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])
    
    pred = np.array(prediction)
    
    q_multi = []
    with open('./results_csv/results_CLL.csv', newline='') as f:
        reader = csv.DictReader(f)
        for q in reader:
            q_multi.append(float(q['q'])
)            
    cll = np.log(np.array(q_multi))
    return auc, acc, ap, precision, recall, HR, cll




def roc_auc_estimator_train(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    if type(pos_edges) == list or type(pos_edges) == np.ndarray:
        for edge in pos_edges:
            prediction.append(reconstructed_adj[edge[0], edge[1]])
            true_label.append(origianl_agjacency[edge[0], edge[1]])

        for edge in negative_edges:
            prediction.append(reconstructed_adj[edge[0], edge[1]])
            true_label.append(origianl_agjacency[edge[0], edge[1]])
    else:
        prediction = list(reconstructed_adj.reshape(-1))
        true_label = list(np.array(origianl_agjacency.todense()).reshape(-1))
    pred = np.array(prediction)
    pred[pred > .5] = 1
    pred[pred < .5] = 0
    pred = pred.astype(int)
    # pred = [1 if x>.5 else 0 for x in prediction]

    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc, acc, ap, cof_mtx


def roc_auc_single(prediction, true_label):
    pred = np.array(prediction)
    pred[pred > .5] = 1
    pred[pred < .5] = 0
    pred = pred.astype(int)

    precision = precision_score(y_pred=pred, y_true=true_label)
    recall = recall_score(y_pred=pred, y_true=true_label)
    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)

    hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:]
    HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])
    
    pred = np.array(prediction)
    cll = np.log((np.concatenate((pred[np.array(true_label) == 1], 1-pred[np.array(true_label) == 0]))))
    
    return auc, acc, ap, precision, recall, HR, cll


# def mask_test_edges(adj, testId, trainId):
#     # Remove diagonal elements
#     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#     # adj.eliminate_zeros()
#     # Check that diag is zero:
#     # assert np.diag(adj.todense()).sum() == 0
#     assert adj.diagonal().sum() == 0

#     A = adj.todense()
#     test_edges = []
#     train_edges = []
#     test_edges_false = []
#     train_edges_false = []
#     for i in range(len(A)):
#         for j in range(len(A[0:])):
#             if i == j:
#                 continue
#             if i in testId and j in testId:
#                 if A[i,j] == 1:
#                     test_edges.append([i,j])                   
#             elif i in trainId and j in trainId:
#                 if A[i,j] == 1:
#                     train_edges.append([i,j])
#     print("Finished Test and Train")


#     while len(test_edges_false) < len(test_edges):
#         for i in range(len(A)):
#             for j in range(len(A[0:])):
#                 if i == j:
#                     continue
#                 if i in testId and j in testId:
#                     if A[i,j] == 0:
#                         test_edges_false.append([i,j])
#     print("Finished Test False")

#     while len(train_edges_false) < len(train_edges):
#         for i in range(len(A)):
#             for j in range(len(A[0:])):
#                 if i == j:
#                     continue
#                 if i in trainId and j in trainId:
#                     if A[i,j] == 0:
#                         train_edges_false.append([i,j])  
#     print("Finished Train False")


# def mask_test_edges(adj, testId, trainId):
#     adj_list_test = sparse.csr_matrix(adj)[testId]
#     adj_list_test = adj_list_test[:, testId]
#     adj_list_test_i , adj_list_test_j = adj_list_test.nonzero()
#     test_edges = []
#     for i in range(len(adj_list_test_i)):
#         test_edges.append([adj_list_test_i[i], adj_list_test_j[i]])

#     test_edges_false = []
#     adj_list_test__false_i , adj_list_test_false_j = sparse.find(adj_list_test==0)[:2]
#     i = 0
#     while len(test_edges_false) < len(test_edges):
#         test_edges_false.append([adj_list_test__false_i[i] , adj_list_test_false_j[i]])
#         i +=1


#     adj_list_train = sparse.csr_matrix(adj)[trainId]
#     adj_list_train = adj_list_train[:, trainId]
#     adj_list_train_i , adj_list_train_j = adj_list_train.nonzero()
#     train_edges = []
#     for i in range(len(adj_list_train_i)):
#         train_edges.append([adj_list_train_i[i], adj_list_train_j[i]])

#     train_edges_false = []
#     adj_list_tain__false_i , adj_list_train_false_j = sparse.find(adj_list_train==0)[:2]
#     i = 0
#     while len(train_edges_false) < len(train_edges):
#         train_edges_false.append([adj_list_tain__false_i[i] , adj_list_train_false_j[i]])
#         i +=1


#     return test_edges_false, test_edges, train_edges_false, train_edges


def mask_test_edges(adj, testId, trainId, validId):
    adj_list = sparse.csr_matrix(adj)
    adj_list_i, adj_list_j = adj_list.nonzero()
    test_edges = []
    train_edges = []
    valid_edges = []
    for i in range(len(adj_list_i)):
        if adj_list_i[i] in testId and adj_list_j[i] in testId:
            test_edges.append([adj_list_i[i], adj_list_j[i]])
        elif adj_list_i[i] in trainId and adj_list_j[i] in trainId:
            train_edges.append([adj_list_i[i], adj_list_j[i]])
        elif adj_list_i[i] in validId and adj_list_j[i] in validId:
            valid_edges.append([adj_list_i[i], adj_list_j[i]])

    test_edges_false = []
    adj_list_false_i, adj_list_false_j = sparse.find(adj_list == 0)[:2]

    while len(test_edges_false) < len(test_edges):
        i = np.random.randint(0, adj_list_false_i.shape[0])
        if adj_list_false_i[i] in testId and adj_list_false_j[i] in testId:
            test_edges_false.append([adj_list_false_i[i], adj_list_false_j[i]])

    train_edges_false = []

    while len(train_edges_false) < len(train_edges):
        i = np.random.randint(0, adj_list_false_i.shape[0])
        if adj_list_false_i[i] in trainId and adj_list_false_j[i] in trainId:
            train_edges_false.append([adj_list_false_i[i], adj_list_false_j[i]])

    valid_edges_false = []

    while len(valid_edges_false) < len(valid_edges):
        i = np.random.randint(0, adj_list_false_i.shape[0])
        if adj_list_false_i[i] in validId and adj_list_false_j[i] in validId:
            valid_edges_false.append([adj_list_false_i[i], adj_list_false_j[i]])

    return test_edges_false, test_edges, train_edges_false, train_edges, valid_edges_false, valid_edges


def kl_pner(m0, m1, s0, s1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.


    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    m0 = m0.detach().numpy()
    m1 = m1.detach().numpy()
    s0 = s0.detach().numpy()
    s1 = s1.detach().numpy()

    # convert std to covariance
    s0 = s0 ** 2
    s1 = s1 ** 2

    n = s1.shape[0]
    d = s1.shape[1]

    N = n * d

    diff = m1 - m0
    diff = diff.reshape(n * d, 1)

    # inverse s1
    s1_inverse = 1 / s1

    # since both s0 and s1 are the elements of the diagonal matrix, their multipication is gonna be the element-wise multipication of s1_inverse and s0
    ss = s1_inverse * s0

    # Trace is same as adding up all the elements on the diagonal
    tr_term = np.sum(ss)

    # det_term: we log of a product can be simplified to sum(log) - sum(log)
    det_term = np.sum(np.log(s1)) - np.sum(np.log(s0))

    # quad_term
    s1_inverse_quad = s1_inverse.reshape(1, n * d)
    quad_term = (diff.T * s1_inverse_quad) @ diff

    return .5 * (tr_term + det_term + quad_term[0][0] - N)


def total_kl(m0, m1, s0, s1, id1, id2):
    m0 = m0.detach().numpy()
    m1 = m1.detach().numpy()
    s0 = s0.detach().numpy()
    s1 = s1.detach().numpy()
    total_res = 0
    torch_res_total = 0
    # node 1
    s0_kl = np.diag(s0[id1] ** 2)
    s1_kl = np.diag(s1[id1] ** 2)
    a = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(m0[id1]), torch.tensor(s0_kl))
    b = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(m1[id1]), torch.tensor(s1_kl))
    kl_res_1 = torch.distributions.kl.kl_divergence(a, b)

    # node 2
    s0_kl = np.diag(s0[id2] ** 2)
    s1_kl = np.diag(s1[id2] ** 2)
    a = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(m0[id2]), torch.tensor(s0_kl))
    b = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(m1[id2]), torch.tensor(s1_kl))
    kl_res_2 = torch.distributions.kl.kl_divergence(a, b)

    return kl_res_1 + kl_res_2

    # for i in range(s0.shape[0]):
    #     total_res += kl_new(m0[i], m1[i], s0[i], s1[i])
    #     s0_kl = np.diag(s0[i]**2)
    #     s1_kl = np.diag(s1[i]**2)
    #     a =torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(m0[i]), torch.tensor(s0_kl))
    #     b = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(m1[i]), torch.tensor(s1_kl))
    #     torch_res_total += torch.distributions.kl.kl_divergence(a, b)

    # print("KL_PNER:    ", kl_pner(m0, m1, s0, s1))
    # print("torch_res_total: ", torch_res_total)
    # print("brand new kl res total: ",total_res )
    # return total_res


def kl_new(m0, m1, s0, s1):
    # convert std to covariance
    s0 = s0 ** 2
    s1 = s1 ** 2

    n = 1
    d = s1.shape[0]

    N = n * d

    diff = m1 - m0
    diff = diff.reshape(n * d, 1)

    # inverse s1
    s1_inverse = 1 / s1

    # since both s0 and s1 are the elements of the diagonal matrix, their multipication is gonna be the element-wise multipication of s1_inverse and s0
    ss = s1_inverse * s0

    # Trace is same as adding up all the elements on the diagonal
    tr_term = np.sum(ss)

    # det_term: we log of a product can be simplified to sum(log) - sum(log)
    det_term = np.sum(np.log(s1)) - np.sum(np.log(s0))

    # quad_term
    s1_inverse_quad = s1_inverse.reshape(1, n * d)
    quad_term = (diff.T * s1_inverse_quad) @ diff

    return .5 * (tr_term + det_term + quad_term[0][0] - N)


def CVAE_loss(m0, m1, s0, s1, pred, labels, id1, id2):
    labels = torch.from_numpy(labels)
    pred = torch.from_numpy(pred)

    pos_weight = torch.true_divide((labels.shape[0] ** 2 - torch.sum(labels)), torch.sum(
        labels))  # addrressing imbalance data problem: ratio between positve to negative instance
    norm = torch.true_divide(labels.shape[0] * labels.shape[0],
                             ((labels.shape[0] * labels.shape[0] - torch.sum(labels) * 2)))

    norm = 1
    posterior_cost = -1 * (norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight))

    kl_term = total_kl(m0, m1, s0, s1, id1, id2)
    # kl_term =  kl_pner(m0, m1, s0, s1)
    # print("posterior cost: ", posterior_cost)
    # print("KL term: ", kl_term)

    return posterior_cost - kl_term


def get_neighbour_prob(rec_adj, idd, neighbour_list):
    rec_adj = torch.sigmoid(rec_adj)
    result = 1
    for neighbour in neighbour_list:
        result *= rec_adj[idd, neighbour]

    return result


def get_metrices(test_edges, org_adj, re_adj):
    return roc_auc_estimator(test_edges, sparse.csr_matrix(torch.sigmoid(re_adj).detach().numpy()),
                             sparse.csr_matrix(org_adj))


# auc_list.append(auc)
# val_acc_list.append(val_acc)
# val_ap_list.append(val_ap)
# precision_list.append(precision)
# recall_list.append(recall)
# HR_list.append(HR)

# return auc_list, val_acc_list, val_ap_list, precision_list,recall_list, HR_list


def run_network(feats, adj, model, targets, sampling_method, is_prior):
    adj = sparse.csr_matrix(adj)
    graph_dgl = dgl.from_scipy(adj)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
    std_z, m_z, z, re_adj = model(graph_dgl, feats, targets, sampling_method, is_prior, train=False)
    return std_z, m_z, z, re_adj


def run_link_encoder_decoder(z_prior, adj, model):
    adj = sparse.csr_matrix(adj)
    graph_dgl = dgl.from_scipy(adj)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
    z, m_z, std_z = model.inference(graph_dgl, z_prior)  # recognition
    re_adj = model.generator(z)
    return std_z, m_z, z, re_adj


def run_link_encoder(z_prior, adj, model):
    adj = sparse.csr_matrix(adj)
    graph_dgl = dgl.from_scipy(adj)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
    z, m_z, std_z = model.inference(graph_dgl, z_prior)  # recognition
    return std_z, m_z, z


def run_feature_encoder(x, model):
    return model.get_z(x, args_kdd.num_of_comunities)


def get_pdf(mean_p, std_p, mean_q, std_q, z, targets):
    
    #print("targets: ",targets)
    n = 1
    d = z.shape[1]
    pdf_all_z_manual_p = 0
    pdf_all_z_manual_q = 0

    pdf_all_z_p = 0
    pdf_all_z_q = 0
    for i in targets:
        # TORCH
        cov_p = np.diag(std_p.detach().numpy()[i] ** 2)
        dist_p = torch.distributions.multivariate_normal.MultivariateNormal(mean_p[i], torch.tensor(cov_p))
        pdf_all_z_p += dist_p.log_prob(z[i]).detach().numpy()

        cov_q = np.diag(std_q.detach().numpy()[i] ** 2)
        dist_q = torch.distributions.multivariate_normal.MultivariateNormal(mean_q[i], torch.tensor(cov_q))
        pdf_all_z_q += dist_q.log_prob(z[i]).detach().numpy()
    return pdf_all_z_p, pdf_all_z_q

        # Manual
        # pdf_all_z_eye_manual += log_pdf(mean[i], torch.ones(z.shape[1]), z[i])
        # pdf_all_z_manual_p += log_pdf(mean_p[i], std_p[i], z[i])
        # pdf_all_z_manual_q += log_pdf(mean_q[i], std_q[i], z[i])

    return pdf_all_z_manual_p, pdf_all_z_manual_q


def log_pdf(m0, s0, z0):
    m0 = m0.detach().numpy()
    s0 = s0.detach().numpy()
    z0 = z0.detach().numpy()

    n = 1
    d = s0.shape[0]

    N = n * d

    diff = z0 - m0
    diff = diff.reshape(n * d, 1)

    # convert std to covariance
    # eps = np.exp(-2)
    # s0= np.ones((n*d))
    # s0 = s0 * eps
    # print("mean s0: ", s0.mean())
    s0 = s0 ** 2
    # inverse s1
    s0_inverse = 1 / s0

    # quad_term
    s0_inverse_quad = s0_inverse.reshape(1, n * d)
    quad_term = (diff.T * s0_inverse_quad) @ diff

    const = np.log(np.power(2 * np.pi, N))

    log_det = np.sum(np.log(s0))

    # coeff = np.sqrt(np.power(2*np.pi,N)*np.prod(s0))

    # coefficient = -1 * np.log(coeff)

    log_pdf = (-1 / 2) * (const + log_det + quad_term)
    # print("const: ", const)
    # print("log_det: ", log_det)
    # print("quad_term: ", quad_term)

    return log_pdf


# def importance_sampling(mp,mq,sp,sq, x, model, adj):

#         print("Importance sampling")
#         s = generated_adj
#         for i in range (0,99):
#             z_0 = run_feature_encoder(x, model)
#             z, m_z, std_z = model.inference(adj, z_0)
#             z = self.dropout(z)
#             generated_adj = self.generator(z)
#             s +=  generated_adj
#         generated_adj = s/100


def make_false_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = sparse.csr_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    index = list(range(train_edges.shape[0]))
    np.random.shuffle(index)
    train_edges_true = train_edges[index[0:num_val]]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])
    # print(test_edges_false)
    # print(val_edges_false)
    # print(test_edges)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    # NOTE: these edge lists only contain single direction of edge!
    return train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_single_link_evidence(adj_recog, idd, neighbours):
    for n in neighbours:
        adj_recog[idd, n] = 0
        adj_recog[n, idd] = 0
    return adj_recog, len(neighbours)
