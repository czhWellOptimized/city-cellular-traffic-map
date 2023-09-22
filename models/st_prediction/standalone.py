import logging
from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch.utils.data import TensorDataset

from datasets.st_datasets import load_dataset
import models.base_models as base_models


def unscaled_metrics(y_pred, y, scaler, name):
    y = scaler.inverse_transform(y.detach().cpu())
    y_pred = scaler.inverse_transform(y_pred.detach().cpu())
    # mse
    mse = ((y_pred - y) ** 2).mean()
    # RMSE
    # rmse = torch.sqrt(mse)
    # MAE
    mae = torch.abs(y_pred - y).mean()
    # MAPE
    mape = torch.abs((y_pred - y) / y).mean()
    return {
        '{}/mse'.format(name): mse.detach(),
        # '{}/rmse'.format(name): rmse.detach(),
        '{}/mae'.format(name): mae.detach(),
        '{}/mape'.format(name): mape.detach()
    }


class NodePredictor(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.base_model = None
        self.base_model_class = getattr(base_models, self.hparams.base_model_name)
        self.setup(None)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.base_model(x, self.global_step) # data, batched_seen

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--hetero_graph', action='store_true')
        return parser

    def prepare_data(self):
        pass

    def setup(self,stage):
        if self.base_model is not None:
            return
        data = load_dataset(dataset_name=self.hparams.dataset)
        self.data = data
        print(len(data)) # 5 feature_scaler,attr_scaler,train,val,test
        # print(data['train']['x']) TxBxNxF  23974*12*207*1
        
        # data的组成
        # feature_scaler,attr_scaler,             train                         ,val,test 
        #                              x,x_attr,y,y_attr,edge_index,edge_attr
        # print(data["train"])
        input_size = self.data['train']['x'].shape[-1] + self.data['train']['x_attr'].shape[-1] # 2
        output_size = self.data['train']['y'].shape[-1] # 1

        assert not ((self.base_model_class is base_models.DCRNNModel) and (self.hparams.hetero_graph))

        if self.base_model_class is base_models.DCRNNModel:
            self.base_model = self.base_model_class(
                adj_mx=to_dense_adj(self.data['train']['edge_index'], edge_attr=self.data['train']['edge_attr']).data.cpu().numpy()[0],
                num_graph_nodes=self.data['train']['x'].shape[2], 
                input_dim=self.data['train']['x'].shape[-1],
                output_dim=output_size,
                seq_len=self.data['train']['x'].shape[1],
                horizon=self.data['train']['y'].shape[1],
                **self.hparams
            )
        else:
            self.base_model = self.base_model_class(
                input_size=input_size,
                output_size=output_size,
                **self.hparams
            )
        self.datasets = {}
        for name in ['train', 'val', 'test']:
            if self.hparams.hetero_graph:
                datalist = [Data(
                    x=self.data[name]['x'][t].permute(1, 0, 2),
                    y=self.data[name]['y'][t].permute(1, 0, 2),
                    x_attr=self.data[name]['x_attr'][t].permute(1, 0, 2),
                    y_attr=self.data[name]['y_attr'][t].permute(1, 0, 2),
                    edge_index=self.data[name]['edge_index'][t],
                    edge_attr=self.data[name]['edge_attr'][t]
                ) for t in range(self.data[name]['x'].shape[0])]
                self.datasets[name] = {
                    'dataset': datalist
                }
            else:
                self.datasets[name] = {
                    'dataset': TensorDataset(self.data[name]['x'], self.data[name]['y'],
                        self.data[name]['x_attr'], self.data[name]['y_attr']),
                    'graph': dict(edge_index=self.data[name]['edge_index'],
                        edge_attr=self.data[name]['edge_attr'])
                }
        # init layer params of DCRNN
        if self.base_model_class is base_models.DCRNNModel:
            temp_dataloader = DataLoader(
                self.datasets['val']['dataset'],
                batch_size=1
            )
            batch = next(iter(temp_dataloader))
            self.validation_step(batch, None)


    def train_dataloader(self):
        return DataLoader(
            self.datasets['train']['dataset'],
            batch_size=self.hparams.batch_size, shuffle=True, num_workers=8 # 多个进程加载数据，所以ps aux中看到的多个进程来自这里
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val']['dataset'],
            batch_size=self.hparams.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test']['dataset'],
            batch_size=self.hparams.batch_size, shuffle=False, num_workers=8
        )

    def configure_optimizers(self):
        if self.base_model_class is base_models.DCRNNModel:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=1e-3, weight_decay=self.hparams.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=[20, 30, 40, 50], gamma=0.1
            )
            return [optimizer], [scheduler]
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, batch_idx):  #  23974  / 128 = 187 .。。分成两个进程执行 每个进程 94次迭代
        if self.hparams.hetero_graph:
            x, y, x_attr, y_attr = batch['x'], batch['y'], batch['x_attr'], batch['y_attr']
            data = batch
        else:
            x, y, x_attr, y_attr = batch
            graph = self.datasets['train']['graph']
            data = dict(
                x=x, x_attr=x_attr, y=y, y_attr=y_attr, 
                edge_index=graph['edge_index'].to(x.device),
                edge_attr=graph['edge_attr'].to(x.device)
            )
        # (torch.Size([128, 12, 207, 1]), <class 'dict'>)
        y_pred = self(data)
        loss = nn.MSELoss()(y_pred, y)

        log = {'train/loss': loss, 'num': y_pred.shape[0]}
        log.update(**unscaled_metrics(y_pred, y, self.data['feature_scaler'], 'train'))
        self.train_step_outputs.append({'loss': loss, 'progress_bar': log, 'log': log})
        
        self.log('czh_all_client_train_loss', log['train/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_train_mse', log['train/mse'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_train_mae', log['train/mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_train_mape', log['train/mape'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        
        
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def on_train_epoch_end(self): # 多次training -> end
        # average all statistics (weighted by sample counts)
        log = {}
        for output in self.train_step_outputs:
            for k in output['log']:
                if k not in log:
                    log[k] = 0
                if k == 'num':
                    log[k] += output['log'][k]
                else:
                    log[k] += (output['log'][k] * output['log']['num'])
        for k in log:
            if k != 'num':
                log[k] = log[k] / log['num']
        log.pop('num')
        
        logging.warning({'type':'train','log_length':len(self.train_step_outputs),'log': log})
        
        self.train_step_outputs.clear()
        
        
        return {'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx): # 3425 / 128 = 27  每个进程13次迭代
        if self.hparams.hetero_graph:
            x, y, x_attr, y_attr = batch['x'], batch['y'], batch['x_attr'], batch['y_attr']
            data = batch
        else:
            x, y, x_attr, y_attr = batch
            graph = self.datasets['val']['graph']
            data = dict(
                x=x, x_attr=x_attr, y=y, y_attr=y_attr, 
                edge_index=graph['edge_index'].to(x.device),
                edge_attr=graph['edge_attr'].to(x.device)
            )
        y_pred = self(data)
        loss = nn.MSELoss()(y_pred, y)
        
        log = {'val/loss': loss, 'num': y_pred.shape[0]}
        log.update(**unscaled_metrics(y_pred, y, self.data['feature_scaler'], 'val'))
        self.validation_step_outputs.append({'loss': loss, 'progress_bar': log, 'log': log})
        
        self.log('czh_all_client_validation_loss', log['val/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_mse', log['val/mse'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_mae', log['val/mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_mape', log['val/mape'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def on_validation_epoch_end(self):
        log = {}
        for output in self.validation_step_outputs: # loss , progress_bar , log
            for k in output['log']: # loss, num, mae, mape , mse
                if k not in log:
                    log[k] = 0 
                if k == 'num':
                    log[k] += output['log'][k]
                else:
                    log[k] += (output['log'][k] * output['log']['num'])
        for k in log:
            if k != 'num':
                log[k] = log[k] / log['num']
        log.pop('num')
        # loss,mae,mape,mse
        logging.warning({'type':'validation','log_length':len(self.validation_step_outputs),'log': log})
        
        
        self.validation_step_outputs.clear()
        
        
        return {'log': log, 'progress_bar': log}

    def test_step(self, batch, batch_idx):
        if self.hparams.hetero_graph:
            x, y, x_attr, y_attr = batch['x'], batch['y'], batch['x_attr'], batch['y_attr']
            data = batch
        else:
            x, y, x_attr, y_attr = batch
            graph = self.datasets['test']['graph']
            data = dict(
                x=x, x_attr=x_attr, y=y, y_attr=y_attr, 
                edge_index=graph['edge_index'].to(x.device),
                edge_attr=graph['edge_attr'].to(x.device)
            )
        y_pred = self(data)
        loss = nn.MSELoss()(y_pred, y)

        log = {'test/loss': loss, 'num': y_pred.shape[0]}
        log.update(**unscaled_metrics(y_pred, y, self.data['feature_scaler'], 'test'))
        self.test_step_outputs.append({'loss': loss, 'progress_bar': log, 'log': log})
        
        self.log('czh_all_client_test_loss', log['test/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_test_mse', log['test/mse'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_test_mae', log['test/mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_test_mape', log['test/mape'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def on_test_epoch_end(self):
        log = {}
        for output in self.test_step_outputs:
            for k in output['log']:
                if k not in log:
                    log[k] = 0
                if k == 'num':
                    log[k] += output['log'][k]
                else:
                    log[k] += (output['log'][k] * output['log']['num'])
        for k in log:
            if k != 'num':
                log[k] = log[k] / log['num']
        log.pop('num')
        self.test_step_outputs.clear()
        
        logging.warning({'type':'test','log_length':len(self.test_step_outputs),'log': log})
        
        return {'log': log, 'progress_bar': log}