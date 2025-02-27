from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch, Data
from models.base_models import WeightedGCN
from models.base_models.GraphNets import GraphNet
import logging

class GRUSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout,
        cl_decay_steps, use_curriculum_learning, gru_num_layers, with_graph_encoding=False,
        *args, **kwargs):
        super().__init__()
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.encoder = nn.GRU(
            input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.with_graph_encoding = with_graph_encoding
        if self.with_graph_encoding:
            self.decoder = nn.GRU(
                input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout
            )
            self.out_net = nn.Linear(2 * hidden_size, output_size)
        else:
            self.decoder = nn.GRU(
                input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
            )
            self.out_net = nn.Linear(hidden_size, output_size)

    def _compute_sampling_threshold(self, batches_seen):
        if self.cl_decay_steps == 0:
            return 0
        else:
            return self.cl_decay_steps / (
                    self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, data, batches_seen, return_encoding=False, graph_encoding=None):
        # B(batch_size) x T(time_slice=12) x N(node_num) x F(特征个数)
        x, x_attr, y, y_attr = data['x'], data['x_attr'], data['y'], data['y_attr']
        if type(data) is Batch: # N x T x F
            logging.warning("Batch:") # 不会执行到这里
            x = x.permute(1, 0, 2).unsqueeze(0) # 
            x_attr = x_attr.permute(1, 0, 2).unsqueeze(0)
            y = y.permute(1, 0, 2).unsqueeze(0)
            y_attr = y_attr.permute(1, 0, 2).unsqueeze(0)
        batch_num, node_num = x.shape[0], x.shape[2]   # 23974*12*207*1 ，由于被分割成batch，所以变成 batch_size*12*207*1
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F (特征此时为2) 对应 seq_length*batch_size*input_size
        _, h_encode = self.encoder(x_input)
        encoder_h = h_encode    # 1*(B x N)*hidden_size  torch.Size([1, 26496, 100])
        # logging.warning(f"FUCK h_encode :{h_encode.shape}") 
        if self.with_graph_encoding:
            h_encode = torch.cat([h_encode, graph_encoding], dim=-1)
        if self.training and (not self.use_curriculum_learning):
            y_input = torch.cat((y, y_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2)
            y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)
            out_hidden, _ = self.decoder(y_input, h_encode)
            out = self.out_net(out_hidden)
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        else:
            last_input = x_input[-1:] # 1 x (B x N) x F
            last_hidden = h_encode #  1 x (B x N) x H
            step_num = y_attr.shape[1] # T 
            out_steps = []
            y_input = y.permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F (特征此时为1)
            y_attr_input = y_attr.permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F (特征此时为1)
            for t in range(step_num):
                out_hidden, last_hidden = self.decoder(last_input, last_hidden) # 1 x (B x N) x H       1*(B x N)* H  
                out = self.out_net(out_hidden) # 1  x (B x N) x output_size
                out_steps.append(out)
                last_input_from_output = torch.cat((out, y_attr_input[t:t+1]), dim=-1) # 1  x (B x N) x (output_size+1 = 2)
                last_input_from_gt = torch.cat((y_input[t:t+1], y_attr_input[t:t+1]), dim=-1) #  1  x (B x N) x 2
                if self.training: # 为什么有时候用到的是真值，有时候用到的是预测值？
                    p_gt = self._compute_sampling_threshold(batches_seen)
                    p = torch.rand(1).item()
                    if p <= p_gt:
                        last_input = last_input_from_gt
                    else:
                        last_input = last_input_from_output
                else:
                    last_input = last_input_from_output
            out = torch.cat(out_steps, dim=0) # T x (B x N) x output_size
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3) # B x T x N x output_size ，
        if type(data) is Batch:
            out = out.squeeze(0).permute(1, 0, 2) # N x T x F
        if return_encoding:
            return out, encoder_h
        else:
            return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--cl_decay_steps', type=int, default=1000)
        parser.add_argument('--use_curriculum_learning', action='store_true')
        parser.add_argument('--gru_num_layers', type=int, default=2)
        return parser


class GRUSeq2SeqWithWeightedGCN(GRUSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_size = kwargs['input_size'] # 2 
        hidden_size = kwargs['hidden_size'] # 64 
        output_size = kwargs['output_size'] # 1
        dropout = kwargs['dropout']
        gru_num_layers = kwargs['gru_num_layers']
        self.decoder = nn.GRU(
            input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.gcn = WeightedGCN(
            input_size=kwargs['hidden_size'],
            hidden_size=kwargs['hidden_size'],
            output_size=kwargs['hidden_size'],
            dropout=kwargs['dropout'],
            inductive=False,
            add_self_loops=False
        )
        self.out_net = nn.Linear(hidden_size * 2, output_size)

    def forward(self, data, batches_seen, return_encoding=False):
        # B x T x N x F
        x, x_attr, y, y_attr = data['x'], data['x_attr'], data['y'], data['y_attr']
        if type(data) is Batch: # N x T x F
            # 不会执行到这里，因为类型是dict
            x = x.permute(1, 0, 2).unsqueeze(0) 
            x_attr = x_attr.permute(1, 0, 2).unsqueeze(0)
            y = y.permute(1, 0, 2).unsqueeze(0)
            y_attr = y_attr.permute(1, 0, 2).unsqueeze(0)
        batch_num, node_num = x.shape[0], x.shape[2]
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F
        _, h_encode = self.encoder(x_input)
        encoder_h = h_encode # layer_num(1) x (B x Ni) x hidden_size

        graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # Ni x B x Layer x hidden_size
        graph_encoding = self.gcn(
            Data(x=graph_encoding, edge_index=data['edge_index'], edge_attr=data['edge_attr'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        )
        graph_encoding = graph_encoding.permute(2, 1, 0, 3).flatten(1, 2) # L x (B x Ni) x F（output_size = intput_size = hidden_size）
        h_encode = torch.cat([h_encode, graph_encoding], dim=-1) # # L x (B x Ni) x 2F

        if self.training and (not self.use_curriculum_learning):
            y_input = torch.cat((y, y_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2)
            y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)
            out_hidden, _ = self.decoder(y_input, h_encode)
            out = self.out_net(out_hidden)
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        else:
            last_input = x_input[-1:] # 1 (B Ni) F
            last_hidden = h_encode # L x (B x Ni) x 2F
            step_num = y_attr.shape[1]
            out_steps = []
            y_input = y.permute(1, 0, 2, 3).flatten(1, 2) # T (B Ni) F
            y_attr_input = y_attr.permute(1, 0, 2, 3).flatten(1, 2) # T (B Ni) F
            for t in range(step_num):
                out_hidden, last_hidden = self.decoder(last_input, last_hidden)
                out = self.out_net(out_hidden) # T x (B x N) x F
                out_steps.append(out)
                last_input_from_output = torch.cat((out, y_attr_input[t:t+1]), dim=-1)
                last_input_from_gt = torch.cat((y_input[t:t+1], y_attr_input[t:t+1]), dim=-1)
                if self.training:
                    p_gt = self._compute_sampling_threshold(batches_seen)
                    p = torch.rand(1).item()
                    if p <= p_gt:
                        last_input = last_input_from_gt
                    else:
                        last_input = last_input_from_output
                else:
                    last_input = last_input_from_output
            out = torch.cat(out_steps, dim=0)
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        if type(data) is Batch:
            out = out.squeeze(0).permute(1, 0, 2) # N x T x F
        if return_encoding:
            return out, encoder_h
        else:
            return out


class GRUSeq2SeqWithGraphNet(GRUSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_size = kwargs['input_size']
        hidden_size = kwargs['hidden_size']
        output_size = kwargs['output_size']
        dropout = kwargs['dropout']
        gru_num_layers = kwargs['gru_num_layers']
        self.gcn_on_server = kwargs['gcn_on_server']
        self.decoder = nn.GRU(
            input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout
        )

        if self.gcn_on_server:
            self.gcn = None
        else:
            self.gcn = GraphNet(
                node_input_size=hidden_size,
                edge_input_size=1,
                global_input_size=hidden_size, 
                hidden_size=256,
                updated_node_size=128,
                updated_edge_size=128,
                updated_global_size=128,
                node_output_size=hidden_size,
                # gn_layer_num=2,
                gn_layer_num=kwargs['gn_layer_num'],
                activation='ReLU', dropout=dropout
            )
        self.out_net = nn.Linear(hidden_size * 2, output_size)

    def _format_input_data(self, data):
        # B x T x N x F
        x, x_attr, y, y_attr = data['x'], data['x_attr'], data['y'], data['y_attr']
        if type(data) is Batch: # N x T x F
            # 不会执行到这里，因为类型是dict
            x = x.permute(1, 0, 2).unsqueeze(0)
            x_attr = x_attr.permute(1, 0, 2).unsqueeze(0)
            y = y.permute(1, 0, 2).unsqueeze(0)
            y_attr = y_attr.permute(1, 0, 2).unsqueeze(0)
        batch_num, node_num = x.shape[0], x.shape[2]
        return x, x_attr, y, y_attr, batch_num, node_num

    def forward_encoder(self, data):
    # BxTxNxF
        x, x_attr, y, y_attr, batch_num, node_num = self._format_input_data(data)
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x Ni) x F
        
        # logging.warning('x_input : %s',str(x_input.shape)) # torch.Size([12,    9936,     2]) 
        #                                                             (seq_len, batch_size, input_size)
        _, h_encode = self.encoder(x_input) 
        # logging.warning('h_encode : %s',str(h_encode.shape)) # torch.Size([1,                        9936,    64])
        #                                                             (num_layers * num_directions, BxN, hidden_size)
        return h_encode # Layer x (B x N) x hidden_size
    # 注意此处分为两个模式，一个是client端训练，那么N=1。一个是server端训练，那么N=207。
    def forward_decoder(self, data, h_encode, batches_seen, return_encoding=False, server_graph_encoding=None): 
        x, x_attr, y, y_attr, batch_num, node_num = self._format_input_data(data)
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x Ni) x F ，需要注意的是此时N=1
        encoder_h = h_encode # Layer x (B x N) x hidden_size

        if self.gcn_on_server:
            graph_encoding = server_graph_encoding # 1 x batch_size x gru_num_layers x hidden_size
        else:
            graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # N x B x L x F
            graph_encoding = self.gcn(
                Data(x=graph_encoding, edge_index=data['edge_index'], edge_attr=data['edge_attr'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            ) # N x B x L x F
        # graph_encoding = 1 x batch_size x gru_num_layers x hidden_size
        graph_encoding = graph_encoding.permute(2, 1, 0, 3).flatten(1, 2) # gru_num_layers x (B x Ni) x hidden_size
        # logging.warning(str(graph_encoding))
        h_encode = torch.cat([h_encode, graph_encoding], dim=-1)  # gru_num_layers x (BxNi) X (hidden_size *2)
        # logging.warning(str(h_encode))
        if self.training and (not self.use_curriculum_learning):
            y_input = torch.cat((y, y_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F
            y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0) 
            out_hidden, _ = self.decoder(y_input, h_encode)
            out = self.out_net(out_hidden)
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        else: # use_curriculum_learning，有时候来自真值，有时候来自预测值
            last_input = x_input[-1:] # 1 x (BxNi) x F
            # logging.warning(str(last_input))
            last_hidden = h_encode # gru_num_layers x (BxNi) X (hidden_size*2)
            step_num = y_attr.shape[1] # T
            out_steps = [] 
            y_input = y.permute(1, 0, 2, 3).flatten(1, 2) # T x (B x Ni) x F
            y_attr_input = y_attr.permute(1, 0, 2, 3).flatten(1, 2)  # Tx(BxNi)xF
            for t in range(step_num):
                # Lx(BxNi)x(2*hidden_size)  Lx(BxNi)x(2*hidden_size)   1x(BxNi)xF ,  gru_num_layersx(BxNi)X(hidden_size*2)
                out_hidden, last_hidden = self.decoder(last_input, last_hidden)  # 可以看到，此处last_input的feature_size 并不影响结果的维度。
                
                out = self.out_net(out_hidden) # Lx(BxNi)xoutput_size
                
                out_steps.append(out)
                last_input_from_output = torch.cat((out, y_attr_input[t:t+1]), dim=-1) # 这里是要求laryer == 1，才能用dim， L(1)x(BxNi)x2
                last_input_from_gt = torch.cat((y_input[t:t+1], y_attr_input[t:t+1]), dim=-1) # 1x(BxNi)x2
                if self.training:
                    p_gt = self._compute_sampling_threshold(batches_seen)
                    p = torch.rand(1).item()
                    if p <= p_gt:
                        last_input = last_input_from_gt
                    else:
                        last_input = last_input_from_output
                else:
                    last_input = last_input_from_output
            out = torch.cat(out_steps, dim=0) # Tx(BxN)xoutput_size
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3) # B x T x Ni x output_size 
        if type(data) is Batch:
            out = out.squeeze(0).permute(1, 0, 2)
        if return_encoding:
            return out, encoder_h
        else:
            return out
    #                                                              1 x batch_size x gru_num_layers x hidden_size
    def forward(self, data, batches_seen, return_encoding=False, server_graph_encoding=None):
        h_encode = self.forward_encoder(data) # Layer x (B x N) x hidden_size
        # logging.warning(str(h_encode))
        return self.forward_decoder(data, h_encode, batches_seen, return_encoding, server_graph_encoding)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = GRUSeq2Seq.add_model_specific_args(parent_parser)
        parser.add_argument('--gn_layer_num', type=int, default=2)
        parser.add_argument('--gcn_on_server', action='store_true')
        return parser