from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MetaLayer
from torch_geometric.data import Batch
from torch_scatter import scatter_add
import logging

class MLP_GN(nn.Module): # 多层感知机
    def __init__(self, input_size, hidden_size, output_size, 
        hidden_layer_num, activation='ReLU', dropout=0.0):
        super().__init__()
        self.net = []
        last_layer_size = input_size
        for _ in range(hidden_layer_num): # input_size -> hidden_size -> ... -> output_size
            self.net.append(nn.Linear(last_layer_size, hidden_size))
            self.net.append(getattr(nn, activation)())
            self.net.append(nn.Dropout(p=dropout))
            last_layer_size = hidden_size
        self.net.append(nn.Linear(last_layer_size, output_size))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class EdgeModel(nn.Module):
    def __init__(self, 
        node_input_size, edge_input_size, global_input_size, 
        hidden_size, edge_output_size, activation, dropout):
        super(EdgeModel, self).__init__()
        edge_mlp_input_size = 2 * node_input_size + edge_input_size + global_input_size
        self.edge_mlp = MLP_GN(edge_mlp_input_size, hidden_size, edge_output_size, 2, activation, dropout)

    def forward(self, src, dest, edge_attr, u, batch): # E*F_e -> E*F_e
        # logging.warning(f"Edge: {src.shape},{dest.shape},{edge_attr.shape},{u.shape},{batch.shape}")
        # source, target: [E, F_x], where E is the number of edges.   src,dst   1722, 48, 1, 64   1722, 48, 1, 64
        # edge_attr: [E, F_e]                             edge x B x L x 1       1722, 48, 1, 1
        # u: [B, F_u], where B is the number of graphs.   graph_number x B x L x global_input_size      1, 48, 1, 64
        # u 表示有B个图，每个图存在特征F_u
        # 
        # batch: [E] with max entry B - 1.               标识各条边都是属于哪一个graph，此处都是graph 0     1722
        
        out = torch.cat([src, dest, edge_attr], -1) # 1722,48,1,129
        if u is not None:
            #print(u[batch].shape) torch.Size([1722, 48, 1, 64])
            out = torch.cat([out, u[batch]], -1) # u[batch] 相当于 E*F，表示每条边所属图的全局特征
            # logging.warning(f"{out.shape}") # 1722, 48, 1, 193     1722, 48, 1, 705
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self,
        node_input_size, edge_input_size, global_input_size,
        hidden_size, node_output_size, activation, dropout):
        super(NodeModel, self).__init__()
        node_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.node_mlp = MLP_GN(node_mlp_input_size, hidden_size, node_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch): # 
        # logging.warning(f"Node: {x.shape},{edge_index.shape},{edge_attr.shape},{u.shape},{batch.shape}")
        # x: [N, F_x], where N is the number of nodes.  N x B x L x hidden_size            207, 48, 1, 64
        # edge_index: [2, E] with max entry N - 1.      src,dst                            2, 1722
        # edge_attr: [E, F_e]                 edge x B x L x 1                             1722, 48, 1, 128
        # u: [B, F_u]                        graph_number x B x L x global_input_size      1, 48, 1, 64
        # batch: [N] with max entry B - 1.   标识各个点都是属于哪一个graph，此处都是graph 0   207
        row, col = edge_index 
        #                          E*F_e     E                     N
        received_msg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        # N,F_x
        out = torch.cat([x, received_msg], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
            # logging.warning(f"{out.shape}") # 207, 48, 1, 256          207, 48, 1, 512
        return self.node_mlp(out)


class GlobalModel(nn.Module):
    def __init__(self,
        node_input_size, edge_input_size, global_input_size,
        hidden_size, global_output_size, activation, dropout):
        super(GlobalModel, self).__init__()
        global_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.global_mlp = MLP_GN(global_mlp_input_size, hidden_size, global_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # logging.warning(f"Global : {x.shape},{edge_index.shape},{edge_attr.shape},{u.shape},{batch.shape}")
        # x: [N, F_x], where N is the number of nodes.   N x B x L x hidden_size            207, 48, 1, 128
        # edge_index: [2, E] with max entry N - 1.       src,dst                            2, 1722
        # edge_attr: [E, F_e]                            edge x B x L x 1                   1722, 48, 1, 128
        # u: [B, F_u]                       graph_number x B x L x global_input_size        1, 48, 1, 64
        # batch: [N] with max entry B - 1.  标识各个点都是属于哪一个graph，此处都是graph 0     207
        row, col = edge_index
        agg_node = scatter_add(x, batch, dim=0)
        agg_edge = scatter_add(scatter_add(edge_attr, col, dim=0, dim_size=x.size(0)), batch, dim=0)
        out = torch.cat([agg_node, agg_edge, u], dim=-1)
        # logging.warning(f"{out.shape}") # 1, 48, 1, 320     1, 17, 1, 448
        return self.global_mlp(out)


class GraphNet(nn.Module): # 64              1             64
    def __init__(self, node_input_size, edge_input_size, global_input_size, 
        # 256
        hidden_size,
        # 128               128                 128
        updated_node_size, updated_edge_size, updated_global_size,
        # 64
        node_output_size,
        # 2
        gn_layer_num, activation, dropout, *args, **kwargs):
        super().__init__()

        self.global_input_size = global_input_size

        self.net = []
        last_node_input_size = node_input_size
        last_edge_input_size = edge_input_size
        last_global_input_size = global_input_size
        # logging.warning("gn_layer_num: %d",gn_layer_num)
        for _ in range(gn_layer_num):
            edge_model = EdgeModel(last_node_input_size, last_edge_input_size, last_global_input_size, hidden_size, updated_edge_size,
                activation, dropout)
            last_edge_input_size += updated_edge_size
            node_model = NodeModel(last_node_input_size, updated_edge_size, last_global_input_size, hidden_size, updated_node_size,
                activation, dropout)
            last_node_input_size += updated_node_size
            global_model = GlobalModel(updated_node_size, updated_edge_size, last_global_input_size, hidden_size, updated_global_size,
                activation, dropout)
            last_global_input_size += updated_global_size
            self.net.append(MetaLayer(
                edge_model, node_model, global_model
            ))
        self.net = nn.ModuleList(self.net)
        self.node_out_net = nn.Linear(last_node_input_size, node_output_size)
    
    def forward(self, data):
        # if not hasattr(data, 'batch'):
        data = Batch.from_data_list([data])
        # N x B x L x hidden_size
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch # batch 是一个一维张量，其中每个元素表示相应的节点属于哪个图形，所以是[207个0]。
        batch = batch.to(x.device)
        
        # logging.warning('batch : %s',str(batch.shape))
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        edge_attr = edge_attr.expand(-1, x.shape[1], x.shape[2], -1) # edge x B x L x 1
        u = x.new_zeros(*([batch[-1] + 1] + list(x.shape[1:-1]) + [self.global_input_size])) # graph_number x B x L x global_input_size
        # u.shape =  1, 48, 1, 64
        for net in self.net:
            updated_x, updated_edge_attr, updated_u = net(x, edge_index, edge_attr, u, batch)
            # logging.warning(str(updated_x)) # nan
            # logging.warning(str(updated_edge_attr)) # nan
            # logging.warning(str(updated_u)) # nan
            x = torch.cat([updated_x, x], dim=-1)
            edge_attr = torch.cat([updated_edge_attr, edge_attr], dim=-1)
            u = torch.cat([updated_u, u], dim=-1)
            # logging.warning(str(x)) # 一半nan
            # logging.warning(str(edge_attr)) # 全部nan
            # logging.warning(str(u)) # 一半nan
            
        node_out = self.node_out_net(x)
        return node_out