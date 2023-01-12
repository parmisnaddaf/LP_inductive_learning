#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:13:06 2021

@author: pnaddaf
"""

import sys
import os
import argparse

import numpy as np
from scipy.sparse import lil_matrix
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import random

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

from dataCenter import *
from utils import *
from models import *
import timeit
#import src.plotter as plotter
#import src.graph_statistics as GS

import classification
import plotter


#%% KDD model
def train_PNModel(dataCenter, features, args, device):
    decoder = args.decoder_type
    encoder = args.encoder_type
    num_of_relations = args.num_of_relations  # diffrent type of relation
    num_of_comunities = args.num_of_comunities  # number of comunities
    batch_norm = args.batch_norm
    DropOut_rate = args.DropOut_rate
    encoder_layers = [int(x) for x in args.encoder_layers.split()]
    split_the_data_to_train_test = args.split_the_data_to_train_test
    epoch_number = args.epoch_number
    negative_sampling_rate = args.negative_sampling_rate
    visulizer_step = args.Vis_step
    PATH = args.mpath
    subgraph_size = args.num_node
    use_feature = args.use_feature
    lr = args.lr
    save_embeddings_to_file = args.save_embeddings_to_file
    is_prior = args.is_prior
    targets = args.targets
    sampling_method = args.sampling_method
    pltr = plotter.Plotter(functions=["loss",  "Accuracy", "Recons Loss", "KL", "AUC"]) 
    synthesis_graphs = {"grid", "community", "lobster", "ego"}
    
    ds = args.dataSet
    if ds in synthesis_graphs:
        synthetic = True
    else: 
        synthetic = False
    
    original_adj_full= torch.FloatTensor(getattr(dataCenter, ds+'_adj_lists')).to(device)

    node_label_full= torch.FloatTensor(getattr(dataCenter, ds+'_labels')).to(device)
    
    # if edge labels exist
    edge_labels = None
    if ds == 'IMDB' or ds == 'ACM' or ds == 'DBLP':
        edge_labels= torch.FloatTensor(getattr(dataCenter, ds+'_edge_labels')).to(device)
    circles = None
    
    
   
    
    # shuffling the data, and selecting a subset of it
    if subgraph_size == -1:
        subgraph_size = original_adj_full.shape[0]
    elemnt = min(original_adj_full.shape[0], subgraph_size)
    indexes = list(range(original_adj_full.shape[0]))
    np.random.shuffle(indexes)
    indexes = indexes[:elemnt]
    original_adj = original_adj_full[indexes, :]
    original_adj = original_adj[:, indexes]
    features = features[indexes]
    if synthetic != True:
        if node_label_full != None:
            node_label = [node_label_full[i] for i in indexes]
        if edge_labels != None:
            edge_labels = edge_labels[indexes, :]
            edge_labels = edge_labels[:, indexes]
        if circles != None:
            shuffles_cir = {}
            for ego_node, circule_lists in circles.items():
                shuffles_cir[indexes.index(ego_node)] = [[indexes.index(x) for x in circule_list] for circule_list in
                                                         circule_lists]
            circles = shuffles_cir
    # Check for Encoder and redirect to appropriate function
    if encoder == "Multi_GCN":
        encoder_model = multi_layer_GCN(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
        # encoder_model = multi_layer_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "Multi_GAT":
        encoder_model = multi_layer_GAT(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "mixture_of_GCNs":
        encoder_model = mixture_of_GCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                        latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)

    elif encoder == "mixture_of_GatedGCNs":
        encoder_model = mixture_of_GatedGCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                             latent_dim=num_of_comunities, layers=encoder_layers, dropOutRate=DropOut_rate)
    elif encoder == "Edge_GCN":
        haveedge = True
        encoder_model = edge_enabled_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)
    # asakhuja End
    else:
        raise Exception("Sorry, this Encoder is not Impemented; check the input args")
    
    # Check for Decoder and redirect to appropriate function
    if decoder == "SBM":
        decoder_model = SBM_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "ML_SBM":
        decoder_model = MultiLatetnt_SBM_decoder(num_of_relations, num_of_comunities, num_of_comunities, batch_norm, DropOut_rate=0.3)
    
    elif decoder == "multi_inner_product":
        decoder_model = MapedInnerProductDecoder([32, 32], num_of_relations, num_of_comunities, batch_norm, DropOut_rate)
    
    elif decoder == "MapedInnerProduct_SBM":
        decoder_model = MapedInnerProduct_SBM([32, 32], num_of_relations, num_of_comunities, batch_norm, DropOut_rate)
    
    elif decoder == "TransE":
        decoder_model = TransE_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "TransX":
        decoder_model = TransX_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "SBM_REL":
        haveedge = True
        decoder_model = edge_enabeled_SBM_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "InnerDot":
        decoder_model = InnerProductDecoder()
    else:
        raise Exception("Sorry, this Decoder is not Impemented; check the input args")
        
        
    feature_encoder_model = feature_encoder(features.view(-1, features.shape[1]), num_of_comunities)  
    if use_feature == False:
        features = torch.eye(features.shape[0])
        features = sp.csr_matrix(features)
    
    
    
    if split_the_data_to_train_test == True:
        trainId = getattr(dataCenter, ds + '_train')
        validId = getattr(dataCenter, ds + '_val')
        testId = getattr(dataCenter, ds + '_test')
        adj_train =  original_adj.cpu().detach().numpy()[trainId, :][:, trainId]
        
        feat_np = features.cpu().data.numpy()
        feat_train = feat_np[trainId, :]
        
        
        
        # adj_train , adj_val, adj_test, feat_train, feat_val, feat_test, train_true, train_false, val_true, val_false= make_test_train_gpu(
        #                 original_adj.cpu().detach().numpy(), features,
        #                 [trainId, validId, testId])
        print('Finish spliting dataset to train and test. ')
    
    
    
    adj_train = sp.csr_matrix(adj_train)
    
    graph_dgl = dgl.from_scipy(adj_train)

    # origianl_graph_statistics = GS.compute_graph_statistics(np.array(adj_train.todense()) + np.identity(adj_train.shape[0]))
    
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops
    
    num_nodes = graph_dgl.number_of_dst_nodes()
    adj_train = torch.tensor(adj_train.todense())  # use sparse man
   
    
    
        
    if (type(feat_train) == np.ndarray):
        feat_train = torch.tensor(feat_train, dtype=torch.float32)
    else:
        feat_train = feat_train
    
    

    # adj_val = sp.csr_matrix(adj_val)
    
    # graph_dgl_val = dgl.from_scipy(adj_val)

    # origianl_graph_statistics = GS.compute_graph_statistics(np.array(adj_train.todense()) + np.identity(adj_train.shape[0]))
    
    # graph_dgl_val.add_edges(graph_dgl_val.nodes(), graph_dgl_val.nodes())  # the library does not add self-loops
    
    # num_nodes_val = graph_dgl_val.number_of_dst_nodes()
    # adj_val = torch.tensor(adj_val.todense())  # use sparse man
    
       
    # if (type(feat_val) == np.ndarray):
    #     feat_val = torch.tensor(feat_val, dtype=torch.float32)
    # else:
    #     feat_val = feat_ 
    

        
    
    
    # randomly select 25% of the test nodes to not be in evidence
    not_evidence = random.sample(list(testId), int(0 * len(testId)))
        
    model = PN_FrameWork(num_of_comunities,
                           encoder=encoder_model,
                           decoder=decoder_model,
                           feature_encoder = feature_encoder_model,
                           not_evidence = not_evidence)  # parameter namimng, it should be dimentionality of distriburion
    

    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(
        adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance
    # pos_wight = torch.tensor(1)
    norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
                             ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))

    for epoch in range(epoch_number):
        # print(epoch)
        model.train()
        # forward propagation by using all nodes
        std_z, m_z, z, reconstructed_adj = model(graph_dgl, feat_train , targets, sampling_method, is_prior, train=True)
        # compute loss and accuracy
        z_kl, reconstruction_loss, acc, val_recons_loss = optimizer_VAE_pn(reconstructed_adj,
                                                                       adj_train + sp.eye(adj_train.shape[0]).todense(),
                                                                       std_z, m_z, num_nodes, pos_wight, norm)
        loss = reconstruction_loss + z_kl
    


        reconstructed_adj = torch.sigmoid(reconstructed_adj).detach().numpy()
        
        model.eval()
        # train_auc, train_acc, train_ap, train_conf = roc_auc_estimator_train(train_true, train_false,
        #                                                       reconstructed_adj, adj_train)
    
    
        # if split_the_data_to_train_test == True:
        #     std_z, m_z, z, reconstructed_adj_val = model(graph_dgl_val, feat_val, is_prior, train=False)
        #     reconstructed_adj_val = torch.sigmoid(reconstructed_adj_val).detach().numpy()
        #     val_auc, val_acc, val_ap, val_conf = roc_auc_estimator_train(val_true, val_false,
        #                                                     reconstructed_adj_val, adj_val)
    
        #     # keep the history to plot
        #     pltr.add_values(epoch, [loss.item(), train_acc,  reconstruction_loss.item(), z_kl, train_auc],
        #                     [None, val_acc, val_recons_loss,None, val_auc  # , val_ap
        #                         ], redraw=False)  # ["Accuracy", "Loss", "AUC", "AP"]
        # else:
        #     # keep the history to plot
        #     pltr.add_values(epoch, [acc, loss.item(), None  # , None
        #                             ],
        #                     [None, None, None  # , None
        #                       ], redraw=False)  # ["Accuracy", "loss", "AUC", "AP"])
    
    
    
        # # Ploting the recinstructed Graph
        # if epoch % visulizer_step == 0:
        #     # pltr.redraw()
        #     print("Val conf:", )
        #     print(val_conf, )
        #     print("Train Conf:")
        #     print(train_conf)
        
        
    
    
    
        model.train()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    
        # print some metrics
        print("Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
            epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
    

    model.eval()
    
    return model, z



