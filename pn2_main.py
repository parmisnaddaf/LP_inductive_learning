#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:10:44 2021

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


#%%  arg setup


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
parser.add_argument('-NofRels', dest="num_of_relations", default=2,
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
parser.add_argument('-v', dest="Vis_step", default=50, help="model learning rate")
parser.add_argument('-modelpath', dest="mpath", default="VGAE_FrameWork_MODEL", type=str,
                    help="The pass to save the learned model")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-s', dest="save_embeddings_to_file", default=True, help="save the latent vector of nodes")

args_kdd = parser.parse_args()

print("")
print("KDD SETING: "+str(args_kdd))



pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])

if torch.cuda.is_available():
	device_id = torch.cuda.current_device()
	print('using device', device_id, torch.cuda.get_device_name(device_id))
else:
    device_id = 'CPU'

device = torch.device(device_id)
print('DEVICE:', device_id)


#%% load config

random.seed(args_kdd.seed)
np.random.seed(args_kdd.seed)
torch.manual_seed(args_kdd.seed)
torch.cuda.manual_seed_all(args_kdd.seed)

# load config file
config = pyhocon.ConfigFactory.parse_file(args_kdd.config)

#%% load data
ds = args_kdd.dataSet
if ds == 'cora':
    dataCenter_sage = DataCenter(config)
    dataCenter_sage.load_dataSet(ds, "graphSage")
    features_sage = torch.FloatTensor(getattr(dataCenter_sage, ds+'_feats')).to(device)
    
    dataCenter_kdd = DataCenter(config)
    dataCenter_kdd.load_dataSet(ds, "KDD")
    features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats')).to(device)
elif ds == 'IMDB' or ds == 'ACM'or ds == 'DBLP':
    dataCenter_kdd = DataCenter(config)
    dataCenter_kdd.load_dataSet(ds, "KDD")
    features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats')).to(device)

    dataCenter_sage = datasetConvert(dataCenter_kdd, ds)
    features_sage = features_kdd



#%%  train inductive_pn
inductive_pn, z_p = helper.train_PNModel(dataCenter_kdd, features_kdd, 
                                      args_kdd, device)




#%% get embedding of PN




# # GET TRAIN EMBEDDINGS
# features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats'))
# trainId = getattr(dataCenter_kdd, ds + '_train')
# labels = getattr(dataCenter_kdd, ds + '_labels')

# adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))
# adj_list_train = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))[trainId]
# adj_list_train = adj_list_train[:, trainId]
# graph_dgl_train = dgl.from_scipy(adj_list_train)
# graph_dgl_train.add_edges(graph_dgl_train.nodes(), graph_dgl_train.nodes())  # the library does not add self-loops



# std_z, m_z, z, re_adj_train = inductive_pn(graph_dgl_train, features_kdd[trainId])
# # re_adj_train = torch.sigmoid(re_adj).detach().numpy()




# GET ALL EMBEDDINGS

adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))
features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats'))
adj_list = adj_list.todense()
adj_list = sparse.csr_matrix(adj_list)
graph_dgl = dgl.from_scipy(adj_list)
graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
std_z_recog, m_z_recog, z_recog, re_adj  = inductive_pn(graph_dgl, features_kdd, train=False)
re_adj = torch.sigmoid(re_adj).detach().numpy()







#%% train classification/prediction model - NN
trainId = getattr(dataCenter_kdd, ds + '_train')
testId = getattr(dataCenter_kdd, ds + '_test')
labels = getattr(dataCenter_kdd, ds + '_labels')










# Link Prediction Task


#get true A
adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))
org_adj = adj_list.toarray()


test_edges_false, test_edges, train_edges_false, train_edges  = mask_test_edges(adj_list, testId, trainId)

# Link Prediction Task
print("=====================================")
print("Result on Link Prediction Task - TRAIN")


# adj_list_train = sparse.csr_matrix(re_adj.detach().numpy())[trainId]
# adj_list_train_re = adj_list_train[:, trainId]
# adj_list_train_re = torch.from_numpy(adj_list_train_re.todense())

# adj_list_train = sparse.csr_matrix(org_adj)[trainId]
# adj_list_train_org = adj_list_train[:, trainId]



auc, val_acc, val_ap, conf_mtrx , precision, recall = roc_auc_estimator(train_edges, train_edges_false, sparse.csr_matrix(re_adj),
                                                        sparse.csr_matrix(org_adj))

print("Train_acc: {:03f}".format(val_acc), " | Train_auc: {:03f}".format(auc), " | Train_AP: {:03f}".format(val_ap)," | Train_precision: {:03f}".format(precision), " | Train_recall: {:03f}".format(recall))
print("Confusion matrix: \n", conf_mtrx)




# testing negative edges
print("=====================================")
print("Result on Link Prediction Task - TEST")

auc, val_acc, val_ap, conf_mtrx , precision, recall = roc_auc_estimator(test_edges, test_edges_false, sparse.csr_matrix(re_adj), sparse.csr_matrix(org_adj))
print("Test_acc: {:03f}".format(val_acc), " | Test_auc: {:03f}".format(auc), " | Test_AP: {:03f}".format(val_ap)," | Test_precision: {:03f}".format(precision), " | Test_recall: {:03f}".format(recall))
print("Confusion matrix: \n", conf_mtrx)


negative_count = 0
adj_list_false_i , adj_list_false_j = sparse.find(sparse.csr_matrix(org_adj)==0)[:2]
total_zeros = len(adj_list_false_i)
for i in range(len(adj_list_false_i)):
    if re_adj[adj_list_false_i[i]][adj_list_false_j[i]] < 0.5:
        negative_count += 1
        
        
    


# testing masking positive edges
auc_list = [auc]
val_acc_list = [val_acc]
val_ap_list = [val_ap]
precision_list = [precision]
recall_list = [recall]
positive_count = 0
total_ones = 0
for idd in testId:
    non_zero_list = org_adj[idd].nonzero()
    for neighbour_id in non_zero_list[0]:
        total_ones += 1
        org_adj_copy = copy.deepcopy(org_adj)
        org_adj_copy[idd,neighbour_id] = 0
        org_adj_copy = sparse.csr_matrix(org_adj_copy)
        graph_dgl = dgl.from_scipy(org_adj_copy)
        graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
        std_z_prior, m_z_prior ,z_prior, re_adj  = inductive_pn(graph_dgl, features_kdd, train=False)
#        kl_output = kl_mvn(m_z_recog, std_z_recog, m_z_prior, std_z_prior)
        # kl_output = torch.distributions.kl.kl_divergence(z_recog.detach().numpy(), z_prior.detach().numpy())
        kl_divergence(m_z_recog,m_z_prior, std_z_recog,std_z_prior)
        re_adj = torch.sigmoid(re_adj).detach().numpy()
        auc, val_acc, val_ap, conf_mtrx , precision, recall= roc_auc_estimator(test_edges, test_edges_false, sparse.csr_matrix(re_adj), sparse.csr_matrix(org_adj))
        auc_list.append(auc)
        val_acc_list.append(val_acc)
        val_ap_list.append(val_ap)
        precision_list.append(precision)
        recall_list.append(recall)
        
        if re_adj[idd][neighbour_id] >= 0.5:
            positive_count += 1
            print("positive_count is: ", positive_count)
            
 


auc_mean = statistics.mean(auc_list)
val_acc_mean = statistics.mean(val_acc_list)
val_ap_mean = statistics.mean(val_ap_list)
precision_mean = statistics.mean(precision_list)
recall_mean = statistics.mean(val_ap_list)
print("val_acc_mean: {:03f}".format(val_acc_mean), " | auc_mean: {:03f}".format(auc_mean),
      " | val_ap_mean: {:03f}".format(val_ap_mean),
      " | precision_mean: {:03f}".format(precision_mean),
      " | recall_mean: {:03f}".format(recall_mean))


hitting_ratio = (positive_count + negative_count) / (total_ones + total_zeros)
print("Hitting ratio is: ", hitting_ratio)








