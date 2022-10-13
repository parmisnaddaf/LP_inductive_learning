#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:51:23 2022

@author: pnaddaf
"""
import sys
import os
import argparse

import numpy as np
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

import copy
from dataCenter import *
from utils import *
from models import *
import plotter as plotter
import graph_statistics as GS
import pn2_helper as helper
import classification
import statistics

# %%  arg setup

##################################################################


parser = argparse.ArgumentParser(description='Inductive KDD')

parser.add_argument('-e', dest="epoch_number", default=100, help="Number of Epochs")
parser.add_argument('--model', type=str, default='KDD')
parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('-num_node', dest="num_node", default=-1, type=str,
                    help="the size of subgraph which is sampled; -1 means use the whule graph")
parser.add_argument('--config', type=str, default='experiments.conf')
parser.add_argument('-decoder_type', dest="decoder_type", default="ML_SBM",
                    help="the decoder type, Either SBM or InnerDot  or TransE or MapedInnerProduct_SBM or multi_inner_product and TransX or SBM_REL")
parser.add_argument('-encoder_type', dest="encoder_type", default="Multi_GCN",
                    help="the encoder type, Either ,mixture_of_GCNs, mixture_of_GatedGCNs , Multi_GCN or Edge_GCN ")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-NofRels', dest="num_of_relations", default=1,
                    help="Number of latent or known relation; number of deltas in SBM")
parser.add_argument('-NofCom', dest="num_of_comunities", default=128,
                    help="Number of comunites, tor latent space dimention; len(z)")
parser.add_argument('-BN', dest="batch_norm", default=True,
                    help="either use batch norm at decoder; only apply in multi relational decoders")
parser.add_argument('-DR', dest="DropOut_rate", default=.3, help="drop out rate")
parser.add_argument('-encoder_layers', dest="encoder_layers", default="64", type=str,
                    help="a list in which each element determine the size of gcn; Note: the last layer size is determine with -NofCom")
parser.add_argument('-lr', dest="lr", default=0.01, help="model learning rate")
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1,
                    help="the rate of negative samples which should be used in each epoch; by default negative sampling wont use")
parser.add_argument('-v', dest="Vis_step", default=5, help="model learning rate")
parser.add_argument('-modelpath', dest="mpath", default="VGAE_FrameWork_MODEL", type=str,
                    help="The pass to save the learned model")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-s', dest="save_embeddings_to_file", default=True, help="save the latent vector of nodes")
parser.add_argument('-CVAE_architecture', dest="CVAE_architecture", default='separate',
                    help="the possible values are sequential, separate, and transfer")
parser.add_argument('-is_prior', dest="is_prior", default=False, help="This flag is used for sampling methods")
parser.add_argument('-disjoint_transductive_inductive', dest="disjoint_transductive_inductive", default=False,
                    help="This flag is used if want to have dijoint transductive and inductive sets")

args_kdd = parser.parse_args()
disjoint_transductive_inductive = args_kdd.disjoint_transductive_inductive

print("")
print("KDD SETING: " + str(args_kdd))

pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print('Using device', device_id, torch.cuda.get_device_name(device_id))
else:
    device_id = 'cpu'

device = torch.device(device_id)

# %% load config

random.seed(args_kdd.seed)
np.random.seed(args_kdd.seed)
torch.manual_seed(args_kdd.seed)
torch.cuda.manual_seed_all(args_kdd.seed)

# load config file
config = pyhocon.ConfigFactory.parse_file(args_kdd.config)

# %% load data
ds = args_kdd.dataSet
if ds == 'cora':
    dataCenter_sage = DataCenter(config)
    dataCenter_sage.load_dataSet(ds, "graphSage")
    features_sage = torch.FloatTensor(getattr(dataCenter_sage, ds + '_feats')).to(device)

    dataCenter_kdd = DataCenter(config)
    dataCenter_kdd.load_dataSet(ds, "KDD")
    features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds + '_feats')).to(device)
elif ds == 'IMDB' or ds == 'ACM' or ds == 'DBLP':
    dataCenter_kdd = DataCenter(config)
    dataCenter_kdd.load_dataSet(ds, "KDD")
    features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds + '_feats')).to(device)

    dataCenter_sage = datasetConvert(dataCenter_kdd, ds)
    features_sage = features_kdd

adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds + '_adj_lists'))
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_edges_true, train_edges_false, ignore_edges_inx, val_edge_idx = mask_test_edges_old(
    adj_list, features_kdd)

# %%  train inductive_pn
inductive_pn, z_p = helper.train_PNModel(dataCenter_kdd, features_kdd,
                                         args_kdd, device)

# Split A into test and train
trainId = getattr(dataCenter_kdd, ds + '_train')
testId = getattr(dataCenter_kdd, ds + '_test')
validId = getattr(dataCenter_kdd, ds + '_val')
labels = getattr(dataCenter_kdd, ds + '_labels')
adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds + '_adj_lists'))
test_edges_false, test_edges, train_edges_false, train_edges, val_edes_false, val_edges = mask_test_edges(adj_list,
                                                                                                          testId,
                                                                                                          trainId,
                                                                                                          validId)

auc_list_multi = []
val_acc_list_multi = []
val_ap_list_multi = []
precision_list_multi = []
recall_list_multi = []
CVAE_list_multi = []
HR_list_multi = []

auc_list_single = []
val_acc_list_single = []
val_ap_list_single = []
precision_list_single = []
recall_list_single = []
HR_list_single = []
CVAE_list_single = []

auc_list_multi_single = []
val_acc_list_multi_single = []
val_ap_list_multi_single = []
precision_list_multi_single = []
recall_list_multi_single = []
HR_list_multi_single = []
CVAE_list_multi_single = []

adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds + '_adj_lists'))
features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds + '_feats'))
org_adj = adj_list.toarray()

prior_only = False
single_link = True
multi_link = False
multi_single_link_bl = False

pred_single_link = []
true_single_link = []
pred_multi_single_link = []
true_multi_single_link = []
pred_multi_link = []
true_multi_link = []

if args_kdd.CVAE_architecture == "separate":
    if disjoint_transductive_inductive:
        adj_train, adj_val, adj_test, feat_train, feat_val, feat_test = make_test_train_gpu(
            org_adj, features_kdd,
            [trainId, validId, testId])
        org_adj = adj_test
        features_kdd = torch.tensor(feat_test)
        train_edges, val_edges, val_edges_false, test_edges, test_edges_false = make_false_edges(org_adj)

    # run recognition separately
    std_z_recog, m_z_recog, z_recog, re_adj_recog = run_network(features_kdd, org_adj, inductive_pn, is_prior=False)
    re_adj_recog_sig = torch.sigmoid(re_adj_recog)
    # run prior network separately
    res = org_adj.nonzero()
    index = np.where(np.isin(res[0], testId))  # only one node of the 2 ends of an edge needs to be in testId
    idd_list = res[0][index]
    neighbour_list = res[1][index]
    sample_list = random.sample(range(0, len(idd_list)), 100)
    for i in sample_list:
        idd = idd_list[i]
        neighbour_id = neighbour_list[i]
        adj_list_copy = copy.deepcopy(org_adj)
        neigbour_prob_single = 1
        if single_link:

            adj_list_copy = copy.deepcopy(org_adj)
            adj_list_copy[idd, neighbour_id] = 0  # find a test edge and set it to 0
            adj_list_copy[neighbour_id, idd] = 0  # find a test edge and set it to 0

            std_z_prior, m_z_prior, z_prior, re_adj_prior = run_network(features_kdd, adj_list_copy, inductive_pn,
                                                                        is_prior=True)
            if prior_only:
                CVAE = CVAE_loss(m_z_prior, m_z_prior, std_z_prior, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            else:
                CVAE = CVAE_loss(m_z_recog, m_z_prior, std_z_recog, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            CVAE_list_single.append(CVAE)
            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            pred_single_link.append(re_adj_prior_sig[idd, neighbour_id].tolist())
            true_single_link.append(org_adj[idd, neighbour_id].tolist())

        if multi_link:
            adj_list_copy = copy.deepcopy(org_adj)
            adj_list_copy[idd, :] = 0  # set all the neigbours to 0
            adj_list_copy[:, idd] = 0  # set all the neigbours to 0

            target_list = [[idd, i] for i in range(adj_list_copy.shape[0])]
            std_z_prior, m_z_prior, z_prior, re_adj_prior = run_network(features_kdd, adj_list_copy, inductive_pn,
                                                                        is_prior=True)

            if prior_only:
                CVAE = CVAE_loss(m_z_prior, m_z_prior, std_z_prior, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            else:
                CVAE = CVAE_loss(m_z_recog, m_z_prior, std_z_recog, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            CVAE_list_multi.append(CVAE)
            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            auc, val_acc, val_ap, conf_mtrx, precision, recall, HR = get_metrices(target_list, org_adj, re_adj_prior)
            auc_list_multi.append(auc)
            val_acc_list_multi.append(val_acc)
            val_ap_list_multi.append(val_ap)
            precision_list_multi.append(precision)
            recall_list_multi.append(recall)
            HR_list_multi.append(HR)
            neighbour_prob_multi_link_list.append(get_neighbour_prob(re_adj_prior, idd, org_adj[idd].nonzero()[
                0]).item())  # this function calculates the prob of all positive edges around idd node

        if multi_single_link_bl:
            for nn in org_adj[idd].nonzero()[0]:
                adj_list_copy = copy.deepcopy(org_adj)
                # set all the neighbours expect the test one to 0 for recognition network
                adj_list_copy, false_count = get_single_link_evidence(adj_list_copy, idd,
                                                                      np.delete(adj_list_copy[idd].nonzero()[0],
                                                                                np.where(adj_list_copy[idd].nonzero()[
                                                                                             0] == nn)))
                std_z_recog, m_z_recog, z_recog, re_adj_recog = run_network(features_kdd, adj_list_copy, inductive_pn,
                                                                            is_prior=False)

                adj_list_copy[idd, nn] = 0  # find a test edge and set it to 0
                adj_list_copy[nn, idd] = 0  # find a test edge and set it to 0
                std_z_prior, m_z_prior, z_prior, re_adj_prior = run_network(features_kdd, adj_list_copy, inductive_pn,
                                                                            is_prior=True)

                if prior_only:
                    CVAE = CVAE_loss(m_z_prior, m_z_prior, std_z_prior, std_z_prior, re_adj_prior.detach().numpy(),
                                     org_adj, idd, neighbour_id).detach().numpy()
                else:
                    CVAE = CVAE_loss(m_z_recog, m_z_prior, std_z_recog, std_z_prior, re_adj_prior.detach().numpy(),
                                     org_adj, idd, neighbour_id).detach().numpy()
                CVAE_list_multi_single.append(CVAE)

                re_adj_prior_sig = torch.sigmoid(re_adj_prior)
                pred_multi_single_link.append(re_adj_prior_sig[idd, nn].tolist())
                true_multi_single_link.append(org_adj[idd, nn].tolist())

            # Get false edges
            std_z_recog, m_z_recog, z_recog, re_adj_recog = run_network(features_kdd, org_adj, inductive_pn,
                                                                        is_prior=False)
            re_adj_recog_sig = torch.sigmoid(re_adj_recog)
            res = np.argwhere(org_adj[idd] == 0)
            np.random.shuffle(res)
            pred_multi_single_link.extend(re_adj_recog_sig[res[:false_count, 0], res[:false_count, 1]].tolist())
            true_multi_single_link.extend(org_adj[res[:false_count, 0], res[:false_count, 1]].tolist())
            auc, val_acc, val_ap, conf_mtrx, precision, recall, HR = roc_auc_single(pred_multi_single_link,
                                                                                    true_multi_single_link)
            auc_list_multi_single.append(auc)
            val_acc_list_multi_single.append(val_acc)
            val_ap_list_multi_single.append(val_ap)
            precision_list_multi_single.append(precision)
            recall_list_multi_single.append(recall)
            HR_list_multi_single.append(HR)

    if single_link:
        false_count = len(pred_single_link)
        res = np.argwhere(org_adj == 0)
        np.random.shuffle(res)
        index = np.where(np.isin(res[:, 0], testId))  # only one node of the 2 ends of an edge needs to be in testId
        test_neg_edges = res[index]
        re_adj_recog_sig = torch.sigmoid(re_adj_recog)
        pred_single_link.extend(
            re_adj_recog_sig[test_neg_edges[:false_count, 0], test_neg_edges[:false_count, 1]].tolist())
        true_single_link.extend(org_adj[test_neg_edges[:false_count, 0], test_neg_edges[:false_count, 1]].tolist())
        auc, val_acc, val_ap, precision, recall, HR = roc_auc_single(pred_single_link, true_single_link)
        auc_list_single.append(auc)
        val_acc_list_single.append(val_acc)
        val_ap_list_single.append(val_ap)
        precision_list_single.append(precision)
        recall_list_single.append(recall)
        HR_list_single.append(HR)

# Print results    


if multi_link:
    auc_mean_multi = statistics.mean(auc_list_multi)
    val_acc_mean_multi = statistics.mean(val_acc_list_multi)
    val_ap_mean_multi = statistics.mean(val_ap_list_multi)
    precision_mean_multi = statistics.mean(precision_list_multi)
    recall_mean_multi = statistics.mean(recall_list_multi)
    HR_mean_multi = statistics.mean(HR_list_multi)

    print("multi link")
    print("auc: ", auc_mean_multi)
    print("acc", val_acc_mean_multi)
    print("ap: ", val_ap_mean_multi)
    print("precision", precision_mean_multi)
    print("recall", recall_mean_multi)
    print("HR", HR_mean_multi)

if multi_single_link_bl:
    auc_mean_multi_single = statistics.mean(auc_list_multi_single)
    val_acc_mean_multi_single = statistics.mean(val_acc_list_multi_single)
    val_ap_mean_multi_single = statistics.mean(val_ap_list_multi_single)
    precision_mean_multi_single = statistics.mean(precision_list_multi_single)
    recall_mean_multi_single = statistics.mean(recall_list_multi_single)
    HR_mean_multi_single = statistics.mean(HR_list_multi_single)

    print("multi link")
    print("auc: ", auc_mean_multi_single)
    print("acc", val_acc_mean_multi_single)
    print("ap: ", val_ap_mean_multi_single)
    print("precision", precision_mean_multi_single)
    print("recall", recall_mean_multi_single)
    print("HR", HR_mean_multi_single)

if single_link:
    auc_mean_single = statistics.mean(auc_list_single)
    val_acc_mean_single = statistics.mean(val_acc_list_single)
    val_ap_mean_single = statistics.mean(val_ap_list_single)
    precision_mean_single = statistics.mean(precision_list_single)
    recall_mean_single = statistics.mean(recall_list_single)
    HR_mean_signle = statistics.mean(HR_list_single)

    print("single link")
    print("auc: ", auc_mean_single)
    print("acc", val_acc_mean_single)
    print("ap: ", val_ap_mean_single)
    print("precision", precision_mean_single)
    print("recall", recall_mean_single)
    print("HR", HR_mean_signle)

# print("CLL single",statistics.mean(neigbour_prob_single_list))
# print("CLL multi",statistics.mean(neighbour_prob_multi_link_list))
