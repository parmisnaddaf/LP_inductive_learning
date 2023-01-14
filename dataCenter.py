import sys
import os

from collections import defaultdict
from scipy.sparse import lil_matrix
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import torch
import json
import pickle
import zipfile


from networkx.readwrite import json_graph
from torch.hub import download_url_to_file


import copy


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data


class DataCenter(object):
    """docstring for DataCenter"""
    def __init__(self, config):
        #super(DataCenter, self).__init__()
        super().__init__()
        self.config = config
        self.test_split = 0.3
        self.val_split = 0.0







    def load_dataSet(self, dataSet='cora', model_name= 'KDD'):
        if model_name == "KDD":
            if dataSet == 'photos' or dataSet == 'computers':
                labels = np.load("./datasets/" + dataSet + "/labels.npy")
                features= np.load("./datasets/" + dataSet + "/x.npy")
                adj = np.load("./datasets/" + dataSet + "/adj.npy")

                test_indexs, val_indexs, train_indexs = self._split_data(features.shape[0])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)

                setattr(self, dataSet+'_feats', features)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj)



            if dataSet == 'citeseer':
                names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
                objects = []
                for i in range(len(names)):
                    with open("./datasets/citeseer/ind.{}.{}".format(dataSet, names[i]), 'rb') as f:
                        if sys.version_info > (3, 0):
                            objects.append(pkl.load(f, encoding='latin1'))
                        else:
                            objects.append(pkl.load(f))

                x, y, tx, ty, allx, ally, graph = tuple(objects)
                test_idx_reorder = parse_index_file("./datasets/citeseer/ind.{}.test.index".format(dataSet))
                test_idx_range = np.sort(test_idx_reorder)


                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
                tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range-min(test_idx_range), :] = tx
                tx = tx_extended
                ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
                ty_extended[test_idx_range-min(test_idx_range), :] = ty
                ty = ty_extended

                features = sp.vstack((allx, tx)).tolil()
                features[test_idx_reorder, :] = features[test_idx_range, :]
                adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

                labels = np.vstack((ally, ty))
                labels[test_idx_reorder, :] = labels[test_idx_range, :]

                test_indexs, val_indexs, train_indexs = self._split_data(features.shape[0])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)

                setattr(self, dataSet+'_feats', features.toarray())
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj.toarray().astype(np.float32))


            if dataSet == 'ppi':
                PPI_PATH = '/local-scratch/parmis/inductive_learning/inductive_learning/ppi'
                PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'  # preprocessed PPI data from Deep Graph Library
                PPI_NUM_INPUT_FEATURES = 50
                PPI_NUM_CLASSES = 121
                if not os.path.exists(PPI_PATH):  # download the first time this is ran
                    os.makedirs(PPI_PATH)

                    # Step 1: Download the ppi.zip (contains the PPI dataset)
                    zip_tmp_path = os.path.join(PPI_PATH, 'ppi.zip')
                    download_url_to_file(PPI_URL, zip_tmp_path)

                    # Step 2: Unzip it
                    with zipfile.ZipFile(zip_tmp_path) as zf:
                        zf.extractall(path=PPI_PATH)
                    print(f'Unzipping to: {PPI_PATH} finished.')

                    # Step3: Remove the temporary resource file
                    os.remove(zip_tmp_path)
                    print(f'Removing tmp file {zip_tmp_path}.')

                # Collect train/val/test graphs here
                edge_index_list = []
                node_features_list = []
                node_labels_list = []

                # Dynamically determine how many graphs we have per split (avoid using constants when possible)
                num_graphs_per_split_cumulative = [0]

                # Small optimization "trick" since we only need test in the playground.py
                splits = ['train', 'valid', 'test']

                for split in splits:
                    # PPI has 50 features per node, it's a combination of positional gene sets, motif gene sets,
                    # and immunological signatures - you can treat it as a black box (I personally have a rough understanding)
                    # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
                    # Note: node features are already preprocessed
                    node_features = np.load(os.path.join(PPI_PATH, f'{split}_feats.npy'))

                    # PPI has 121 labels and each node can have multiple labels associated (gene ontology stuff)
                    # SHAPE = (NS, 121)
                    node_labels = np.load(os.path.join(PPI_PATH, f'{split}_labels.npy'))

                    # Graph topology stored in a special nodes-links NetworkX format
                    nodes_links_dict = json_read(os.path.join(PPI_PATH, f'{split}_graph.json'))
                    # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
                    # The reason I use a NetworkX's directed graph is because we need to explicitly model both directions
                    # because of the edge index and the way GAT implementation #3 works
                    collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
                    # For each node in the above collection, ids specify to which graph the node belongs to
                    graph_ids = np.load(os.path.join(PPI_PATH, F'{split}_graph_id.npy'))
                    num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))

                    # Split the collection of graphs into separate PPI graphs
                    for graph_id in range(np.min(graph_ids), 15):
                        mask = graph_ids == graph_id# find the nodes which belong to the current graph (identified via id)
                        graph_node_ids = np.asarray(mask).nonzero()[0]
                        graph = collection_of_graphs.subgraph(graph_node_ids)  # returns the induced subgraph over these nodes
                        print(f'Loading {split} graph {graph_id} to CPU. '
                              f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')

                        # shape = (2, E) - where E is the number of edges in the graph
                        # Note: leaving the tensors on CPU I'll load them to GPU in the training loop on-the-fly as VRAM
                        # is a scarcer resource than CPU's RAM and the whole PPI dataset can't fit during the training.
                        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).transpose(0, 1).contiguous()
                        edge_index = edge_index - edge_index.min()  # bring the edges to [0, num_of_nodes] range
                        edge_index_list.append(edge_index)
                        # shape = (N, 50) - where N is the number of nodes in the graph
                        node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float))
                        # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
                        node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float))

                adj = np.zeros((node_features_list[13].shape[0], node_features_list[13].shape[0]))
                # for i in range(edge_index_list[13][0].shape[0]):
                #     adj[edge_index_list[13][0][i].item()][edge_index_list[13][1][i].item()] = 1
                adj[edge_index_list[13][0], edge_index_list[13][1]] = 1

                features = node_features_list[13].numpy()
                labels = node_labels_list[13].numpy()

                test_indexs, val_indexs, train_indexs = self._split_data(features.shape[0])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)

                setattr(self, dataSet+'_feats', features)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj)


            if dataSet == 'cora':
                cora_content_file = './datasets/cora/cora.content'
                cora_cite_file = './datasets/cora/cora.cites'

                with open(cora_content_file) as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
                id_list = []
                for x in content:
                    x = x.split()
                    id_list.append(int(x[0]))
                id_list = list(set(id_list))
                old_to_new_dict = {}
                for idd in id_list:
                        old_to_new_dict[idd] = len(old_to_new_dict.keys())

                with open(cora_cite_file) as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
                edge_list = []
                for x in content:
                    x = x.split()
                    edge_list.append([old_to_new_dict[int(x[0])] , old_to_new_dict[int(x[1])]])

                all_nodes = set()
                for pair in edge_list:
                    all_nodes.add(pair[0])
                    all_nodes.add(pair[1])

                adjancy_matrix = lil_matrix((len(all_nodes), len(all_nodes)))

                for pair in edge_list:
                    adjancy_matrix[pair[0],pair[1]] = 1
                    adjancy_matrix[pair[1],pair[0]] = 1

                feat_data = []
                labels = [] # label sequence of node
                node_map = {} # map node to Node_ID
                label_map = {} # map label to Label_ID
                with open(cora_content_file) as fp:
                    for i,line in enumerate(fp):
                        info = line.strip().split()
                        feat_data.append([float(x) for x in info[1:-1]])
                        node_map[info[0]] = i
                        if not info[-1] in label_map:
                            label_map[info[-1]] = len(label_map)
                        labels.append(label_map[info[-1]])
                feat_data = np.asarray(feat_data)
                labels = np.asarray(labels, dtype=np.int64)

                test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adjancy_matrix.toarray())

            if dataSet == "IMDB":
                obj = []

                adj_file_name = "./datasets/IMDB/edges.pkl"

                with open(adj_file_name, 'rb') as f:
                    obj.append(pkl.load(f))

                # merging diffrent edge type into a single adj matrix
                adj = lil_matrix(obj[0][0].shape)
                for matrix in obj[0]:
                    adj +=matrix

                matrix = obj[0]
                edge_labels = matrix[0] + matrix[1]
                edge_labels += (matrix[2] + matrix[3])*2

                node_label= []
                in_1 = matrix[0].indices.min()
                in_2 = matrix[0].indices.max()+1
                in_3 = matrix[2].indices.max()+1
                node_label.extend([0 for i in range(in_1)])
                node_label.extend([1 for i in range(in_1,in_2)])
                node_label.extend([2 for i in range(in_2, in_3)])

                obj = []
                with open("./datasets/IMDB/node_features.pkl", 'rb') as f:
                    obj.append(pkl.load(f))
                feature = sp.csr_matrix(obj[0])
                feature = sp.csr_matrix(obj[0])

                index = 9000
                test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)

                setattr(self, dataSet+'_feats', feature[:index].toarray())
                setattr(self, dataSet+'_labels', np.array(node_label[:index]))
                setattr(self, dataSet+'_adj_lists', adj[:index,:index].toarray())
                setattr(self, dataSet+'_edge_labels', edge_labels[:index].toarray())

            if dataSet == "ACM":
                obj = []
                adj_file_name = "./datasets/ACM/edges.pkl"
                with open(adj_file_name, 'rb') as f:
                        obj.append(pkl.load(f))

                adj = sp.csr_matrix(obj[0][0].shape)
                for matrix in obj:
                    nnz = matrix[0].nonzero() # indices of nonzero values
                    for i, j in zip(nnz[0], nnz[1]):
                        adj[i,j] = 1
                        adj[j,i] = 1
                    #adj +=matrix[0]

                # to fix the bug on running GraphSAGE
                adj = adj.toarray()
                for i in range(len(adj)):
                    if sum(adj[i, :]) == 0:
                        idx = np.random.randint(0, len(adj))
                        adj[i,idx] = 1
                        adj[idx,i] = 1

                edge_labels = matrix[0] + matrix[1]
                edge_labels += (matrix[2] + matrix[3])*2

                node_label= []
                in_1 = matrix[0].indices.min()
                in_2 = matrix[0].indices.max()+1
                in_3 = matrix[2].indices.max()+1
                node_label.extend([0 for i in range(in_1)])
                node_label.extend([1 for i in range(in_1,in_2)])
                node_label.extend([2 for i in range(in_2, in_3)])


                obj = []
                with open("./datasets/ACM/node_features.pkl", 'rb') as f:
                    obj.append(pkl.load(f))
                feature = sp.csr_matrix(obj[0])


                index = -1
                test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)

                setattr(self, dataSet+'_feats', feature[:index].toarray())
                setattr(self, dataSet+'_labels',np.array(node_label[:index]))
                setattr(self, dataSet+'_adj_lists', adj[:index,:index])
                setattr(self, dataSet+'_edge_labels', edge_labels[:index,:index].toarray())

            elif dataSet == "DBLP":

                obj = []

                adj_file_name = "./datasets/DBLP/edges.pkl"


                with open(adj_file_name, 'rb') as f:
                        obj.append(pkl.load(f))

                # merging diffrent edge type into a single adj matrix
                adj = sp.csr_matrix(obj[0][0].shape)
                for matrix in obj[0]:
                    adj +=matrix

                matrix = obj[0]
                edge_labels = matrix[0] + matrix[1]
                edge_labels += (matrix[2] + matrix[3])*2

                node_label= []
                in_1 = matrix[0].nonzero()[0].min()
                in_2 = matrix[0].nonzero()[0].max()+1
                in_3 = matrix[3].nonzero()[0].max()+1
                matrix[0].nonzero()
                node_label.extend([0 for i in range(in_1)])
                node_label.extend([1 for i in range(in_1,in_2)])
                node_label.extend([2 for i in range(in_2, in_3)])


                obj = []
                with open("./datasets/node_features.pkl", 'rb') as f:
                    obj.append(pkl.load(f))
                feature = sp.csr_matrix(obj[0])


                index = -1000
                test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)

                setattr(self, dataSet+'_feats', feature[:index].toarray())
                setattr(self, dataSet+'_labels', np.array(node_label[:index]))
                setattr(self, dataSet+'_adj_lists', adj[:index,:index].toarray())
                setattr(self, dataSet+'_edge_labels', edge_labels[:index].toarray())



    def _split_data(self, num_nodes, test_split = 0.2, val_split = 0.1):
        np.random.seed(123)
        rand_indices = np.random.permutation(num_nodes)

        test_size = int(num_nodes * test_split)
        val_size = int(num_nodes * val_split)
        train_size = num_nodes - (test_size + val_size)
        # print(num_nodes, train_size, val_size, test_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size+val_size)]
        train_indexs = rand_indices[(test_size+val_size):]

        return test_indexs, val_indexs, train_indexs
