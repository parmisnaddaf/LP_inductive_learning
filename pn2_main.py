#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:51:23 2022

@author: pnaddaf
"""

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


parser.add_argument('-e', dest="epoch_number", default=200, help="Number of Epochs")
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
parser.add_argument('-v', dest="Vis_step", default=50, help="model learning rate")
parser.add_argument('-modelpath', dest="mpath", default="VGAE_FrameWork_MODEL", type=str,
                    help="The pass to save the learned model")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-s', dest="save_embeddings_to_file", default=True, help="save the latent vector of nodes")
parser.add_argument('-CVAE_architecture', dest="CVAE_architecture", default='separate', help="the possible values are sequential, separate, and transfer")

args_kdd = parser.parse_args()

print("")
print("KDD SETING: "+str(args_kdd))



pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])

if torch.cuda.is_available():
	device_id = torch.cuda.current_device()
	print('using device', device_id, torch.cuda.get_device_name(device_id))
else:
    device_id = 'cpu'

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


# Split A into test and train
trainId = getattr(dataCenter_kdd, ds + '_train')
testId = getattr(dataCenter_kdd, ds + '_test')
labels = getattr(dataCenter_kdd, ds + '_labels')
adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))
test_edges_false, test_edges, train_edges_false, train_edges  = mask_test_edges(adj_list, testId, trainId)







auc_list = []
val_acc_list = []
val_ap_list = []
precision_list = []
recall_list = []
CVAE_list = []
nodes_list= []




adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds+'_adj_lists'))
features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds+'_feats'))
org_adj = adj_list.toarray()
xx = 0
count = 0
if args_kdd.CVAE_architecture == "separate":
    # run recognition separately
    std_z_recog , m_z_recog , z_recog , re_adj_recog = run_network(features_kdd, org_adj, inductive_pn)
    get_matrices(test_edges, test_edges_false, org_adj, re_adj_recog, auc_list, val_acc_list, val_ap_list, precision_list,recall_list)
    re_adj_recog_sig = torch.sigmoid(re_adj_recog)
    # run prior network separately
    for idd in testId:
        non_zero_list = org_adj[idd].nonzero()
        for neighbour_id in non_zero_list[0]:
            nodes_list.append((idd, neighbour_id))
            xx += 1
            if xx % 50 == 0:
                print(xx)
            # if xx == 1000:
            #     break
            adj_list_copy = copy.deepcopy(org_adj)
            adj_list_copy[idd,neighbour_id] = 0 # find a test edge anD set it to 0
            std_z_prior , m_z_prior , z_prior , re_adj_prior = run_network(features_kdd, adj_list_copy, inductive_pn)
            get_matrices(test_edges, test_edges_false, org_adj, re_adj_prior, auc_list, val_acc_list, val_ap_list, precision_list,recall_list)
            CVAE = CVAE_loss(m_z_recog,m_z_prior, std_z_recog,std_z_prior, re_adj_recog.detach().numpy(), org_adj, idd, neighbour_id).detach().numpy()
            CVAE_list.append(CVAE)
            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            if re_adj_recog_sig[idd,neighbour_id].item() >= re_adj_prior_sig[idd,neighbour_id].item():
                count += 1

        # if xx == 1000:
        #     break
if args_kdd.CVAE_architecture == "sequential":
    # the ouput of the prior is going to be the input of the recognition
    for idd in testId:
        non_zero_list = org_adj[idd].nonzero()
        for neighbour_id in non_zero_list[0]:
            nodes_list.append((idd, neighbour_id))
            # xx += 1
            # print(xx)
            # if xx == 100:
            #     break
            adj_list_copy = copy.deepcopy(org_adj)
            # adj_list_copy[idd,neighbour_id] = 0 # find a test edge and set it to 0
            std_z_prior , m_z_prior , z_prior , re_adj_prior = run_network(features_kdd, adj_list_copy, inductive_pn) # prior
            get_matrices(test_edges, test_edges_false, org_adj, re_adj_prior , auc_list, val_acc_list, val_ap_list, precision_list,recall_list)
            
            # z_prior2 = run_feature_enocder(features_kdd, inductive_pn)
            
            std_z_recog , m_z_recog , z_recog , re_adj_recog  = run_link_enocder(z_prior, adj_list, inductive_pn)
            # zeros_needed = features_kdd.shape[1] - z_prior.shape[1]
            # z_prior_masked = np.hstack((z_prior.detach().numpy(), np.zeros((z_prior.detach().numpy().shape[0], zeros_needed)))) #add zeros at the end of z_prior to make it be the shape of features_kdd

            # std_z_recog , m_z_recog , z_recog , re_adj_recog = run_network(torch.tensor(z_prior_masked.astype(np.float32)), adj_list, inductive_pn) # recognition
            get_matrices(test_edges, test_edges_false, org_adj, re_adj_recog , auc_list, val_acc_list, val_ap_list, precision_list,recall_list)     
            CVAE_list.append(CVAE_loss(m_z_recog,m_z_prior, std_z_recog,std_z_prior, re_adj_recog.detach().numpy(), org_adj, idd, neighbour_id).detach().numpy())

        # if xx == 100:
        #     break


# Print results    
auc_mean = statistics.mean(auc_list)
val_acc_mean = statistics.mean(val_acc_list)
val_ap_mean = statistics.mean(val_ap_list)
precision_mean = statistics.mean(precision_list)
recall_mean = statistics.mean(recall_list)
CVAE_mean = np.mean(CVAE_list)
P_Y_given_E =  np.exp(np.mean(CVAE_list))

print("=====================================")
print("Result on Link Prediction Task - Between TEST edges")
print("val_acc_mean: {:03f}".format(val_acc_mean), " | auc_mean: {:03f}".format(auc_mean),
      " | val_ap_mean: {:03f}".format(val_ap_mean),
      " | precision_mean: {:03f}".format(precision_mean),
      " | recall_mean: {:03f}".format(recall_mean),
       " | CVAE_mean: {:03f}".format(CVAE_mean),
       " | P_Y_given_E: {:03f}".format(P_Y_given_E))