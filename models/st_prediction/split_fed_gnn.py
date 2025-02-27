import os
from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from datasets.st_datasets import load_dataset
import models.base_models as base_models
from models.st_prediction.standalone import unscaled_metrics
from models.base_models.GraphNets import GraphNet
import logging

import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import os
from torch_geometric.utils import dense_to_sparse

import sys


class SplitFedNodePredictorClient(nn.Module):
    def __init__(self, base_model_name, optimizer_name,
        train_dataset, val_dataset, test_dataset, feature_scaler,
        sync_every_n_epoch, lr, weight_decay, batch_size, client_device, start_global_step,
        *args, **kwargs):
        super().__init__()
        self.base_model_name = base_model_name
        self.optimizer_name = optimizer_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_scaler = feature_scaler
        self.sync_every_n_epoch = sync_every_n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.base_model_kwargs = kwargs
        self.device = client_device

        self.base_model_class = getattr(base_models, self.base_model_name)
        self.init_base_model(None)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        if self.val_dataset:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.val_dataloader = self.train_dataloader
        if self.test_dataset:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_dataloader = self.train_dataloader

        self.global_step = start_global_step

    def forward(self, x, server_graph_encoding): 
        return self.base_model(x, self.global_step, server_graph_encoding=server_graph_encoding) # global_step = batch_seen

    def init_base_model(self, state_dict):
        self.base_model = self.base_model_class(**self.base_model_kwargs).to(self.device)
        if state_dict is not None:
            self.base_model.load_state_dict(state_dict)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def local_train(self, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.train()
        with torch.enable_grad():
            for epoch_i in range(self.sync_every_n_epoch):
                num_samples = 0
                epoch_log = defaultdict(lambda : 0.0)
                for batch in self.train_dataloader:
                    x, y, x_attr, y_attr, server_graph_encoding = batch # B 1 L H  ，应该修改成 B Ni L H 
                    # after : Sequence_length（batch_size） x 1 x gru_num_layers x hidden_size ，但是由于分batch了，所以sequence_length 此时变成batch_size
                    server_graph_encoding = server_graph_encoding.permute(1, 0, 2, 3) # 1 x batch_size x gru_num_layers x hidden_size
                    # logging.warning(str(server_graph_encoding))
                    x = x.to(self.device) if (x is not None) else None
                    y = y.to(self.device) if (y is not None) else None
                    x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                    y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                    server_graph_encoding = server_graph_encoding.to(self.device)
                    
                    data = dict(
                        x=x, x_attr=x_attr, y=y, y_attr=y_attr
                    )
                    y_pred = self(data, server_graph_encoding) # 调用 forward(self, x, server_graph_encoding)  B x T x Ni x output_size
                    # logging.warning(str(y_pred.device))
                    # logging.warning(str(y.device))
                    
                    loss = nn.MSELoss()(y_pred, y) # B x T x Ni x output_size 
                    # logging.warning("train,  y:  ,y_pred:   ,loss: %s", str(loss))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    num_samples += x.shape[0] * x.shape[2]
                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 'train')
                    epoch_log['train/loss'] += loss.detach() * x.shape[0] * x.shape[2]
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0] * x.shape[2]
                    self.global_step += 1
                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        state_dict = self.base_model.to('cpu').state_dict()
        epoch_log['num_samples'] = num_samples
        epoch_log['global_step'] = self.global_step
        epoch_log = dict(**epoch_log)

        return {
            'state_dict': state_dict, 'log': epoch_log
        }

    def local_eval(self, dataloader, name, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.eval()
        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda : 0.0)
            for batch in dataloader:
                x, y, x_attr, y_attr, server_graph_encoding = batch # x : B T N F 
                server_graph_encoding = server_graph_encoding.permute(1, 0, 2, 3)
                x = x.to(self.device) if (x is not None) else None
                y = y.to(self.device) if (y is not None) else None
                x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                server_graph_encoding = server_graph_encoding.to(self.device)
                # logging.warning(str(server_graph_encoding))
                data = dict(
                    x=x, x_attr=x_attr, y=y, y_attr=y_attr
                )
                y_pred = self(data, server_graph_encoding)
                loss = nn.MSELoss()(y_pred, y)
                # logging.warning("eval,  y:  %s,y_pred: %s  ,loss: %s", str(y),str(y_pred),str(loss))
                num_samples += x.shape[0] * x.shape[2]
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler, name)
                # logging.warning(metrics)
                # sys.exit()
                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0] * x.shape[2]
                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0] * x.shape[2]
            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)
        
        return {'log': epoch_log}

    def local_validation(self, state_dict_to_load):
        return self.local_eval(self.val_dataloader, 'val', state_dict_to_load)

    def local_test(self, state_dict_to_load):
        return self.local_eval(self.test_dataloader, 'test', state_dict_to_load)

    @staticmethod
    def client_local_execute(device, state_dict_to_load, order, **hparams_list): # 只支持mp_worker=1
        if (type(device) is str) and (device.startswith('cuda:')):
            cuda_id = int(device.split(':')[1])
            # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
            device = torch.device('cuda:{}'.format(cuda_id))
            # device = torch.device('cuda:0')
            torch.cuda.set_device(device)
        elif type(device) is torch.device:
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')

        client = SplitFedNodePredictorClient(client_device=device, **hparams_list)
        if order == 'train':
            res = client.local_train(state_dict_to_load)
        elif order == 'val':
            res = client.local_validation(state_dict_to_load)
        elif order == 'test':
            res = client.local_test(state_dict_to_load)
        else:
            del client
            raise NotImplementedError()
        del client
        return res

    # def cuda(self, device, manual_control=False):
    #     if not manual_control:
    #         return super().to('cpu')
    #     else: # only move to GPU when manual_control explicitly set!
    #         return super().cuda(device)

    # def to(self, device, manual_control=False, *args, **kwargs):
    #     if not manual_control:
    #         return super().to('cpu')
    #     else:
    #         return super().to(*args, **kwargs)


# def fednodepredictorclient_


class SplitFedNodePredictor(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.base_model = None
        self.setup(None)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        # return self.base_model(x)
        raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--server_batch_size', type=int, default=8)
        parser.add_argument('--sync_every_n_epoch', type=int, default=5)
        parser.add_argument('--server_epoch', type=int, default=5)
        parser.add_argument('--mp_worker_num', type=int, default=8)
        parser.add_argument('--server_gn_layer_num', type=int, default=2)
        parser.add_argument('--clusters',type=int,default=20)
        return parser

    def prepare_data(self):
        pass
    
    def get_clusters(self):
        current_directory = os.getcwd()
        # 读取数据
        df = pd.read_csv(current_directory + '/preprocess/valid_topology.csv')
        clusters = self.hparams.clusters
        print("cluster:",clusters)
        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=clusters, random_state=0,n_init='auto').fit(df[['lon', 'lat']])
        # 将聚类标签添加到数据框中
        df['cluster'] = kmeans.labels_
        # print(df)
        
        client_idx_list = []
        adj_list = []
        
        # 打印每个类的基站id和中心基站的id
        count =0
        for i in range(clusters):
            print(f"Cluster {i}:")
            cluster_df = df[df['cluster']==i]
            bs_ids = cluster_df['bs'].tolist() # 得到该类中的所有bs id
            # 使用a中的索引从df中提取所需的基站经纬度
            df_tmp = df.iloc[cluster_df.index]
            # 将经纬度转换为弧度
            df_tmp.loc[:, ['lon', 'lat']] = np.radians(df_tmp[['lon', 'lat']])
            # print(df_tmp)
            # 计算haversine距离
            distances = haversine_distances(df_tmp[['lat', 'lon']], df_tmp[['lat', 'lon']]) * 6371000
            # print(distances)
            # 创建邻接矩阵
            distances_matrix = pd.DataFrame(distances, index=df_tmp['bs'], columns=df_tmp['bs'])
            distances_matrix_np = distances_matrix.values
            # 邻接矩阵单位是米
            # print(distances_matrix_np,end='\n\n')

            normalized_k = 0.1
            distances = distances_matrix_np[~np.isinf(distances_matrix_np)].flatten()
            std = distances.std() + 1e-6 # 防止单个基站，出现除0的情况
            adj_mx = np.exp(-np.square(distances_matrix / std))
            # Make the adjacent matrix symmetric by taking the max.
            # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
            # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
            # 这行代码将邻接矩阵中所有小于阈值normalized_k的元素设置为零。这样可以增加邻接矩阵的稀疏性，也就是说，只有那些距离较大（即原始距离矩阵中的值较大）的基站之间才会有连接。
            adj_mx[adj_mx < normalized_k] = 0
            # print("adj_mx.shape:",adj_mx.shape)
            # print(type(adj_mx))
            print("Base stations len:", len(bs_ids))
            print("Base stations :", bs_ids)
            tmp = list(cluster_df.index)
            count += len(tmp)
            # print(tmp)
            client_idx_list.append(tmp)
            # print(adj_mx)
            my_array = np.array(adj_mx)
            my_tensor = torch.from_numpy(my_array).float()
            print("adj_mx_ts.shape:",my_tensor.shape)
            # print(my_tensor)
            adj_list.append(dense_to_sparse(my_tensor))
            
            # sys.exit()
        logging.warning(count)
        return client_idx_list,adj_list
        
    def setup(self, stage):
        # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
        if self.base_model is not None:
            return
        data = load_dataset(dataset_name=self.hparams.dataset)
        self.data = data
        # Each node (client) has its own model and optimizer
        # Assigning data, model and optimizer for each client
        num_clients = data['train']['x'].shape[2] # 23974 * 12 * 207 * 1      ，这里需要做聚类，改成 num_clusters   
        # num_clients = 1 DEBUG用
        input_size = self.data['train']['x'].shape[-1] + self.data['train']['x_attr'].shape[-1] # 2
        output_size = self.data['train']['y'].shape[-1] # 1
        client_params_list = []
        
        client_idx_list,adj_list = self.get_clusters()
        num_clients = len(client_idx_list)
        # sys.exit()
        client=[]
        for x in client_idx_list:
            client.append(torch.tensor(x))
        # B x T x N x F 
        for client_i in range(num_clients):
            client_datasets = {}
            for name in ['train', 'val', 'test']:
                # print("shape:",torch.index_select(data[name]['x'],2,client[client_i]).shape)
                client_datasets[name] = TensorDataset( # B T Ni F
                    torch.index_select(data[name]['x'],2,client[client_i]),
                    torch.index_select(data[name]['y'],2,client[client_i]),
                    torch.index_select(data[name]['x_attr'],2,client[client_i]),
                    torch.index_select(data[name]['y_attr'],2,client[client_i]),
                    torch.zeros(len(client[client_i]), data[name]['x'].shape[0], self.hparams.gru_num_layers, self.hparams.hidden_size).float().permute(1, 0, 2, 3) # default_server_graph_encoding
                    # after : B x 1 x gru_num_layers x hidden_size ，需要改成 B Ni L H
                )
            client_params = {}
            client_params.update(
                optimizer_name='Adam',
                train_dataset=client_datasets['train'],
                val_dataset=client_datasets['val'],
                test_dataset=client_datasets['test'],
                feature_scaler=self.data['feature_scaler'],
                input_size=input_size,
                output_size=output_size,
                start_global_step=0,
                own_bs_index = client[client_i],
                own_adj = adj_list[client_i],
                **self.hparams
            )
            client_params_list.append(client_params)
            # print("client length: ",client_params['own_bs_index']) 
            # print("client length: ",len(client_params['own_bs_index']))
        self.client_params_list = client_params_list
        print("client para list size:",len(self.client_params_list)) # 20 
        self.base_model = getattr(base_models, self.hparams.base_model_name)(input_size=input_size, output_size=output_size, **self.hparams)
        self.gcn = GraphNet(
            node_input_size=self.hparams.hidden_size,
            edge_input_size=1,
            global_input_size=self.hparams.hidden_size,
            hidden_size=256,
            updated_node_size=128,
            updated_edge_size=128,
            updated_global_size=128,
            node_output_size=self.hparams.hidden_size,
            # gn_layer_num=2,
            gn_layer_num=self.hparams.server_gn_layer_num,
            activation='ReLU', dropout=self.hparams.dropout
        )
        self.server_optimizer = getattr(torch.optim, 'Adam')(self.gcn.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.server_datasets = {}
        for name in ['train', 'val', 'test']:
            self.server_datasets[name] = TensorDataset(
                self.data[name]['x'], self.data[name]['y'], 
                self.data[name]['x_attr'], self.data[name]['y_attr']
            )

    def _train_server_gcn_with_agg_clients(self, device):
        # here we assume all clients are aggregated! Simulate running on clients with the aggregated copy on server
        # this only works when (1) clients are aggregated and (2) no optimization on client models
        self.base_model.to(device)
        self.gcn.to(device)
        server_train_dataloader = DataLoader(self.server_datasets['train'], batch_size=self.hparams.server_batch_size, shuffle=True)

        global_step = self.client_params_list[0]['start_global_step']
        with torch.enable_grad():
            self.base_model.train()
            self.gcn.train()
            for epoch_i in range(self.hparams.server_epoch + 1): 
                if epoch_i == self.hparams.server_epoch:
                    server_train_dataloader = DataLoader(self.server_datasets['train'], batch_size=self.hparams.server_batch_size, shuffle=False)
                for batch in server_train_dataloader:
                    x, y, x_attr, y_attr = batch
                    x = x.to(device) if (x is not None) else None
                    y = y.to(device) if (y is not None) else None
                    x_attr = x_attr.to(device) if (x_attr is not None) else None
                    y_attr = y_attr.to(device) if (y_attr is not None) else None
                    
                    if 'selected' in self.data['train']:
                        train_mask = self.data['train']['selected'].flatten()
                        x, y, x_attr, y_attr = x[:, :, train_mask, :], y[:, :, train_mask, :], x_attr[:, :, train_mask, :], y_attr[:, :, train_mask, :]

                    for i,c in enumerate(self.client_params_list): # 每个client分别进行 GNN 处理    
                        tmp = c['own_bs_index'].to(device)
                        data = dict(
                            x=torch.index_select(x,2,tmp).to(device), 
                            x_attr=torch.index_select(x_attr,2,tmp).to(device),
                            y=torch.index_select(y,2,tmp).to(device), 
                            y_attr=torch.index_select(y_attr,2,tmp).to(device)
                        )
                        h_encode = self.base_model.forward_encoder(data) # Layer x (B x N) x hidden_size ，此处N=全部节点
                        batch_num, node_num = data['x'].shape[0], data['x'].shape[2]  # B , N
                        graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # N x B x L x hidden_size
                                                
                        # logging.warning('train before graph_encoding : %s',str(graph_encoding)) # torch.Size([3, 36, 1, 64])
                        # logging.warning('train edgeindex : %s',str(c['own_adj'][0])) # torch.Size([3, 36, 1, 64])
                        # logging.warning('train edge_attr : %s',str(c['own_adj'][1])) # torch.Size([3, 36, 1, 64])

                        graph_encoding = self.gcn(
                            Data(x=graph_encoding, 
                            edge_index = c['own_adj'][0].to(graph_encoding.device),
                            edge_attr=c['own_adj'][1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(graph_encoding.device))
                        ) # Ni x B x L x F
                        
                        if torch.isnan(graph_encoding).any():
                            logging.warning("train after graph_encoding is nan: %s",str(graph_encoding)) # torch.Size([195, 36, 1, 64]) 
                        
                        if epoch_i == self.hparams.server_epoch:
                            name = 'train'
                            keyname = '{}_dataset'.format(name)
                            c.update({
                                keyname: TensorDataset(
                                    torch.index_select(self.data[name]['x'],2,c['own_bs_index']),
                                    torch.index_select(self.data[name]['y'],2,c['own_bs_index']),
                                    torch.index_select(self.data[name]['x_attr'],2,c['own_bs_index']),
                                    torch.index_select(self.data[name]['y_attr'],2,c['own_bs_index']),
                                    graph_encoding.detach().clone().cpu().permute(1, 0, 2, 3)
                                )
                            })
                        else:
                            y_pred = self.base_model.forward_decoder(
                                data, h_encode, batches_seen=global_step, return_encoding=False, server_graph_encoding=graph_encoding
                            ) # B x T x N x output_size
                            loss = nn.MSELoss()(y_pred, data['y'])
                            self.server_optimizer.zero_grad()
                            loss.backward()
                            self.server_optimizer.step()
                            global_step += 1
        # update global step for all clients
        for client_params in self.client_params_list:
            client_params.update(start_global_step=global_step)

    def _eval_server_gcn_with_agg_clients(self, name, device):
        assert name in ['val', 'test']
        self.base_model.to(device)
        self.gcn.to(device)
        server_dataloader = DataLoader(self.server_datasets[name], batch_size=self.hparams.server_batch_size, shuffle=False)
        
        with torch.no_grad():
            self.base_model.eval()
            self.gcn.eval()
            for batch in server_dataloader:
                x, y, x_attr, y_attr = batch
                x = x.to(device) if (x is not None) else None
                y = y.to(device) if (y is not None) else None
                x_attr = x_attr.to(device) if (x_attr is not None) else None
                y_attr = y_attr.to(device) if (y_attr is not None) else None
                
                for i,c in enumerate(self.client_params_list): # 每个client分别进行 GNN 处理
                    tmp = c['own_bs_index'].to(device)
                    data = dict(
                        x=torch.index_select(x,2,tmp).to(device), 
                        x_attr=torch.index_select(x_attr,2,tmp).to(device),
                        y=torch.index_select(y,2,tmp).to(device), 
                        y_attr=torch.index_select(y_attr,2,tmp).to(device)
                    )
                    # logging.warning("eval:x.shape:!!!!! %s ",str(data['x'].shape)) # B T N F
                    
                    h_encode = self.base_model.forward_encoder(data) # Layer x (Bi x N) x hidden_size
                    batch_num, node_num = data['x'].shape[0], data['x'].shape[2] # server_batch_size 48 。195, 236, 78, 263, 33, 223, 42
                    graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # Ni x B x L x hidden_size

                    graph_encoding = self.gcn(
                        Data(x=graph_encoding, 
                        edge_index = c['own_adj'][0].to(graph_encoding.device),
                        edge_attr=c['own_adj'][1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(graph_encoding.device))
                    ) # Ni x B x L x F
                    if torch.isnan(graph_encoding).any():
                       logging.warning("eval after graph_encoding is nan: %s",str(graph_encoding)) # torch.Size([195, 36, 1, 64])     
                    
                    keyname = '{}_dataset'.format(name)
                    c.update({
                        keyname: TensorDataset(
                            torch.index_select(self.data[name]['x'],2,c['own_bs_index']),
                            torch.index_select(self.data[name]['y'],2,c['own_bs_index']),
                            torch.index_select(self.data[name]['x_attr'],2,c['own_bs_index']),
                            torch.index_select(self.data[name]['y_attr'],2,c['own_bs_index']),
                            graph_encoding.detach().clone().cpu().permute(1, 0, 2, 3)
                        )
                    })

    def train_dataloader(self):
        # return a fake dataloader for running the loop
        return DataLoader([0,])

    def val_dataloader(self):
        return DataLoader([0,])

    def test_dataloader(self):
        return DataLoader([0,])

    def configure_optimizers(self):
        return None

    def backward(self, trainer):
        return None

    def training_step(self, batch, batch_idx):
        # 1. train locally and collect uploaded local train results
        local_train_results = []
        server_device = next(self.gcn.parameters()).device
        self.base_model.to('cpu')
        self.gcn.to('cpu')
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                if 'selected' in self.data['train']:
                    if self.data['train']['selected'][client_i, 0].item() is False:
                        continue
                # logging.warning(client_i)
                local_train_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(self.base_model.state_dict()), 'train', **client_params)
                local_train_results.append(local_train_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            temp_client_params_list = []
            for client_i, client_params in enumerate(self.client_params_list):
                if 'selected' in self.data['train']:
                    if self.data['train']['selected'][client_i, 0].item() is False:
                        continue
                temp_client_params_list.append(client_params)
            for worker_i, client_params in enumerate(np.array_split(temp_client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                # device_name = 'cpu'
                local_train_results.append(pool.apply_async(SplitFedNodePredictorClient.client_local_execute, args=(
                    device_name, deepcopy(self.base_model.state_dict()), 'train', client_params)))
            pool.close()
            pool.join()
            local_train_results = list(map(lambda x: x.get(), local_train_results))
            local_train_results = list(itertools.chain.from_iterable(local_train_results))
        # update global steps for all clients
        for ltr, client_params in zip(local_train_results, self.client_params_list):
            client_params.update(start_global_step=ltr['log']['global_step'])
        # 2. aggregate (optional? kept here to save memory, otherwise need to store 1 model for each node)
        agg_local_train_results = self.aggregate_local_train_results(local_train_results)
        # 2.1. update aggregated weights
        if agg_local_train_results['state_dict'] is not None:
            self.base_model.load_state_dict(agg_local_train_results['state_dict']) # value是一个字典
        # TODO: 3. train GNN on server in split learning way (optimize server-side params only)
        # client_local_execute, return_encoding
        # run decoding on all clients, run backward on all clients
        # run backward on server-side GNN and optimize GNN
        # TODO: 4. run forward on updated GNN to renew server_graph_encoding
        # logging.warning("client process success !")
        self._train_server_gcn_with_agg_clients(server_device)
        # logging.warning("server gcn process success !")
        agg_log = agg_local_train_results['log']
        log = agg_log
        self.train_step_outputs.append({'loss': torch.tensor(0).float(), 'progress_bar': log, 'log': log})
         
        self.log('czh_all_client_train_loss', log['train/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_train_rmse', log['train/rmse'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_train_mae', log['train/mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_train_mape', log['train/mape'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        
        return {'loss': torch.tensor(0).float(), 'progress_bar': log, 'log': log}

    def aggregate_local_train_results(self, local_train_results):
        return {
            'state_dict': self.aggregate_local_train_state_dicts(
                [ltr['state_dict'] for ltr in local_train_results]
            ),
            'log': self.aggregate_local_logs(
                [ltr['log'] for ltr in local_train_results]
            )
        }

    def aggregate_local_train_state_dicts(self, local_train_state_dicts):
        raise NotImplementedError()

    def aggregate_local_logs(self, local_logs, selected=None): # 可以通过修改num_samples的方式，不修改此处
        agg_log = deepcopy(local_logs[0])
        if selected is not None:
            agg_log_t = deepcopy(local_logs[0])
            agg_log_i = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            if selected is not None:
                agg_log_t[k] = 0
                agg_log_i[k] = 0
            for local_log_idx, local_log in enumerate(local_logs):
                if k == 'num_samples':
                    agg_log[k] += local_log[k]
                else:
                    agg_log[k] += local_log[k] * local_log['num_samples']
                if selected is not None:
                    is_trans = selected[local_log_idx, 0].item()
                    if is_trans:
                        if k == 'num_samples':
                            agg_log_t[k] += local_log[k]
                        else:
                            agg_log_t[k] += local_log[k] * local_log['num_samples']
                    else:
                        if k == 'num_samples':
                            agg_log_i[k] += local_log[k]
                        else:
                            agg_log_i[k] += local_log[k] * local_log['num_samples']
        for k in agg_log:
            if k != 'num_samples':
                agg_log[k] /= agg_log['num_samples']
                if selected is not None:
                    agg_log_t[k] /= agg_log_t['num_samples']
                    agg_log_i[k] /= agg_log_i['num_samples']
        if selected is not None:
            for k in agg_log_t:
                agg_log[k + '_trans'] = agg_log_t[k]
            for k in agg_log_i:
                agg_log[k + '_induc'] = agg_log_i[k]
        return agg_log

    def on_train_epoch_end(self):
        # already averaged!
        log = self.train_step_outputs[0]['log']
        log.pop('num_samples')
        self.train_step_outputs.clear()
        logging.warning({'log': log, 'progress_bar': log})
        return {'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        server_device = next(self.gcn.parameters()).device
        self._eval_server_gcn_with_agg_clients('val', server_device) # TODO 为啥 test和validation的server gcn在前面，而train在后面
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        self.base_model.to('cpu')
        self.gcn.to('cpu')
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_val_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(self.base_model.state_dict()), 'val', **client_params)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(np.array_split(self.client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                # device_name = 'cpu'
                local_val_results.append(pool.apply_async(
                    SplitFedNodePredictorClient.client_local_execute, args=(
                        device_name, deepcopy(self.base_model.state_dict()), 'val', client_params)
                ))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(itertools.chain.from_iterable(local_val_results))
        self.base_model.to(server_device)
        self.gcn.to(server_device)
        # 2. aggregate
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        self.validation_step_outputs.append({'progress_bar': log, 'log': log})
        
        self.log('czh_all_client_validation_loss', log['val/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_rmse', log['val/rmse'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_mae', log['val/mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_mape', log['val/mape'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True)
        
        return {'progress_bar': log, 'log': log}

    def on_validation_epoch_end(self):
        log = self.validation_step_outputs[0]['log']
        log.pop('num_samples')
        self.validation_step_outputs.clear()
        logging.warning({'log': log, 'progress_bar': log})
        return {'log': log, 'progress_bar': log}

    def test_step(self, batch, batch_idx):
        server_device = next(self.gcn.parameters()).device
        self._eval_server_gcn_with_agg_clients('test', server_device)
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        self.base_model.to('cpu')
        self.gcn.to('cpu')
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_val_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(self.base_model.state_dict()), 'test', **client_params)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(np.array_split(self.client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                # device_name = 'cpu'
                local_val_results.append(pool.apply_async(
                    SplitFedNodePredictorClient.client_local_execute, args=(
                        device_name, deepcopy(self.base_model.state_dict()), 'test', client_params)
                ))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(itertools.chain.from_iterable(local_val_results))
        self.base_model.to(server_device)
        self.gcn.to(server_device)
        # 2. aggregate
        # separate seen and unseen nodes if necessary
        if 'selected' in self.data['train']:
            log = self.aggregate_local_logs([x['log'] for x in local_val_results], self.data['train']['selected'])
        else:
            log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        self.test_step_outputs.append({'progress_bar': log, 'log': log})
        
        self.log('czh_all_client_test_loss', log['test/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_test_rmse', log['test/rmse'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_test_mae', log['test/mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_test_mape', log['test/mape'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        
        return {'progress_bar': log, 'log': log}

    def test_epoch_end(self):
        log = self.test_step_outputs[0]['log']
        log.pop('num_samples')
        self.test_step_outputs.clear()
        logging.warning({'log': log, 'progress_bar': log})
        return {'log': log, 'progress_bar': log}


class SplitFedAvgNodePredictor(SplitFedNodePredictor):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

    def aggregate_local_train_state_dicts(self, local_train_state_dicts): # TODO
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:
            agg_state_dict[k] = 0
            for ltsd in local_train_state_dicts:
                agg_state_dict[k] += ltsd[k]
            agg_state_dict[k] /= len(local_train_state_dicts)
        return agg_state_dict