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

import logging


class FedNodePredictorClient(nn.Module):
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

    def forward(self, x):
        return self.base_model(x, self.global_step)

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
            for epoch_i in range(self.sync_every_n_epoch): # 每一个client实际进行多少次epoch，需要设置sync_every_n_epoch=1
                num_samples = 0
                epoch_log = defaultdict(lambda : 0.0)
                for batch in self.train_dataloader:
                    x, y, x_attr, y_attr = batch
                    x = x.to(self.device) if (x is not None) else None
                    y = y.to(self.device) if (y is not None) else None
                    x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                    y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                    data = dict(
                        x=x, x_attr=x_attr, y=y, y_attr=y_attr
                    )
                    y_pred = self(data)
                    loss = nn.MSELoss()(y_pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    num_samples += x.shape[0] # B T N F         
                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 'train')
                    epoch_log['train/loss'] += loss.detach() * x.shape[0]
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0]
                    self.global_step += 1
                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        state_dict = self.base_model.to('cpu').state_dict()          
        epoch_log['num_samples'] = num_samples       # 最后一个epoch中的num_samples 
        # logging.warning(f"FUCK samples , should be 32xxx : {num_samples}" )
        epoch_log['global_step'] = self.global_step  # 所有epoch的batch总和。
        epoch_log = dict(**epoch_log)
        # logging.warning(f"FUCK epoch_log length :it should be 6, : {len(epoch_log)}")
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
                x, y, x_attr, y_attr = batch
                x = x.to(self.device) if (x is not None) else None
                y = y.to(self.device) if (y is not None) else None
                x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                data = dict(
                    x=x, x_attr=x_attr, y=y, y_attr=y_attr
                )
                y_pred = self(data)
                loss = nn.MSELoss()(y_pred, y)
                num_samples += x.shape[0]
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler, name)
                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0]
                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]
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
    def client_local_execute(device, order, **hparams_list):
        # logging.warning(f" 111FUCK list type : {type(hparams_list)}")
        
        torch.cuda.empty_cache()
        if (type(device) is str) and (device.startswith('cuda:')): # 只有设置 --mp_worker_num > 1才会执行到这里
            cuda_id = int(device.split(':')[1])
            # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
            device = torch.device('cuda:{}'.format(cuda_id))
            # device = torch.device('cuda:0')
        elif type(device) is torch.device:
            pass
        else:
            device = torch.device('cpu')
        torch.cuda.set_device(device)
        
        state_dict_to_load = hparams_list['state_dict_to_load']
        client = FedNodePredictorClient(client_device=device, **hparams_list) # 构造函数，此时已经将数据包含进去了
        if order == 'train':
            res = client.local_train(state_dict_to_load)
        elif order == 'val':
            res = client.local_validation(state_dict_to_load)
        elif order == 'test':
            res = client.local_test(state_dict_to_load)
        else:
            del client
            torch.cuda.empty_cache()
            raise NotImplementedError()
        del client
        torch.cuda.empty_cache()
        return res

        
    @staticmethod
    def client_local_execute2(device, order, hparams_list):
        # logging.warning(f" 222FUCK list type : {type(hparams_list)}")
        
        torch.cuda.empty_cache()
        if (type(device) is str) and (device.startswith('cuda:')): # 只有设置 --mp_worker_num > 1才会执行到这里
            cuda_id = int(device.split(':')[1])
            # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
            device = torch.device('cuda:{}'.format(cuda_id))
            # logging.warning(f"FUCK device type is : {type(device)}")
            # device = torch.device('cuda:0')
        elif type(device) is torch.device:
            pass
        else:
            device = torch.device('cpu')
        torch.cuda.set_device(device)
        
        res_list = []
        for hparams in hparams_list:
            state_dict_to_load = hparams['state_dict_to_load']
            client = FedNodePredictorClient(client_device=device, **hparams) # 构造函数，此时已经将数据包含进去了
            if order == 'train':
                res = client.local_train(state_dict_to_load)
            elif order == 'val':
                res = client.local_validation(state_dict_to_load)
            elif order == 'test':
                res = client.local_test(state_dict_to_load)
            else:
                del client
                torch.cuda.empty_cache()
                raise NotImplementedError()
            del client
            torch.cuda.empty_cache()
            res_list.append(res)
        return res_list

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


class FedNodePredictor(LightningModule):
    def __init__(self, hparams, identical_agg_model, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.identical_agg_model = identical_agg_model
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
        parser.add_argument('--sync_every_n_epoch', type=int, default=5)
        parser.add_argument('--mp_worker_num', type=int, default=8)
        return parser

    def prepare_data(self):
        pass

    def setup(self, stage):
        # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
        if self.base_model is not None:
            return
        data = load_dataset(dataset_name=self.hparams.dataset)
        self.data = data
        # Each node (client) has its own model and optimizer
        # Assigning data, model and optimizer for each client
        num_clients = data['train']['x'].shape[2]   # T x B x N x F
        # num_clients = 2 # DEBUG用
        
        input_size = self.data['train']['x'].shape[-1] + self.data['train']['x_attr'].shape[-1]
        output_size = self.data['train']['y'].shape[-1]
        client_params_list = []
        for client_i in range(num_clients):
            client_datasets = {}
            for name in ['train', 'val', 'test']:
                client_datasets[name] = TensorDataset(
                    data[name]['x'][:, :, client_i:client_i+1, :],
                    data[name]['y'][:, :, client_i:client_i+1, :],
                    data[name]['x_attr'][:, :, client_i:client_i+1, :],
                    data[name]['y_attr'][:, :, client_i:client_i+1, :]
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
                **self.hparams
            )
            client_params_list.append(client_params)
        self.client_params_list = client_params_list

        if self.identical_agg_model:
            self.base_model = getattr(base_models, self.hparams.base_model_name)(input_size=input_size, output_size=output_size, **self.hparams)
        else:
            self.base_model = []
            for idx in range(num_clients):
                self.base_model.append(
                    getattr(base_models, self.hparams.base_model_name)(input_size=input_size, output_size=output_size, **self.hparams)
                )
                self.base_model[-1].load_state_dict(deepcopy(self.base_model[0].state_dict()))
            self.base_model = nn.ModuleList(self.base_model)

    def _get_copied_ith_model(self, idx):
        if self.identical_agg_model:
            return deepcopy(self.base_model.state_dict())
        else:
            return deepcopy(self.base_model[idx].state_dict())

    def train_dataloader(self):
        # return a fake dataloader for running the loop
        # dataset size=1,batch_size=1,so one epoch = one iteration,so it's a fake dataloader,so one epoch = call training_step one time
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
        for client_i, client_params in enumerate(self.client_params_list):
            client_params.update(state_dict_to_load=self._get_copied_ith_model(client_i))
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                if 'selected' in self.data['train']:
                    if self.data['train']['selected'][client_i, 0].item() is False:
                        continue
                # logging.warning(client_i)
                local_train_result = FedNodePredictorClient.client_local_execute(batch.device, 'train', **client_params)
                # local_train_result = {
                #     'state_dict': state_dict, 'log': epoch_log
                # }
                local_train_results.append(local_train_result) # [ ret1 , ret2]
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
                local_train_results.append(pool.apply_async(FedNodePredictorClient.client_local_execute2, args=(
                    device_name, 'train', client_params)))
            pool.close()
            pool.join()
            local_train_results = list(map(lambda x: x.get(), local_train_results)) # [[],[],[],...[]] 
            local_train_results = list(itertools.chain.from_iterable(local_train_results)) # [ , , , ... ,]
        # update global steps for all clients
        for ltr, client_params in zip(local_train_results, self.client_params_list):
            client_params.update(start_global_step=ltr['log']['global_step'])   # global_step：每个client总共跑的batch个数
        # 2. aggregate
        agg_local_train_results = self.aggregate_local_train_results(local_train_results)
        # 3. update aggregated weights
        # if agg_local_train_results['state_dict'] is not None:
        if self.identical_agg_model:
            self.base_model.load_state_dict(agg_local_train_results['state_dict']) 
        else:
            for idx in range(len(self.base_model)):
                self.base_model[idx].load_state_dict(agg_local_train_results['state_dict'][idx])
        agg_log = agg_local_train_results['log'] # agg_log 字典，所有client最后一个epoch的平均数据
        # 3. send aggregated weights to all clients
        # if self.last_round_weight is not None:
        #     for client_i, client in enumerate(self.clients):
        #         client.load_weights(deepcopy(agg_state_dict))
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

    def aggregate_local_logs(self, local_logs, selected=None):
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
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        for client_i, client_params in enumerate(self.client_params_list):
            client_params.update(state_dict_to_load=self._get_copied_ith_model(client_i))
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                # logging.warning(f"val :{len(client_params)}  {type(client_params)}")
                local_val_result = FedNodePredictorClient.client_local_execute(batch.device, 'val', **client_params)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(np.array_split(self.client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                # logging.warning(f"val :{len(client_params)}  {type(client_params)}")
                local_val_results.append(pool.apply_async(
                    FedNodePredictorClient.client_local_execute2, args=(
                        device_name, 'val', client_params)
                ))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(itertools.chain.from_iterable(local_val_results))
        # 2. aggregate
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        self.validation_step_outputs.append({'progress_bar': log, 'log': log})
        
        self.log('czh_all_client_validation_loss', log['val/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_rmse', log['val/rmse'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_mae', log['val/mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        self.log('czh_all_client_validation_mape', log['val/mape'], on_step=True, on_epoch=True, prog_bar=True, logger=True ,sync_dist=True) 
        
        return {'progress_bar': log, 'log': log}

    def on_validation_epoch_end(self):
        # already averaged!
        log = self.validation_step_outputs[0]['log']
        log.pop('num_samples')
        self.validation_step_outputs.clear()
        logging.warning({'log': log, 'progress_bar': log})
        return {'log': log, 'progress_bar': log}

    def test_step(self, batch, batch_idx):
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        for client_i, client_params in enumerate(self.client_params_list):
            client_params.update(state_dict_to_load=self._get_copied_ith_model(client_i))
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_val_result = FedNodePredictorClient.client_local_execute(batch.device, 'test', **client_params)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(np.array_split(self.client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                local_val_results.append(pool.apply_async(
                    FedNodePredictorClient.client_local_execute2, args=(
                        device_name, 'test', client_params)
                ))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(itertools.chain.from_iterable(local_val_results))
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
        # already averaged!
        log = self.test_step_outputs[0]['log']
        log.pop('num_samples')
        self.test_step_outputs.clear()
        logging.warning({'log': log, 'progress_bar': log})
        return {'log': log, 'progress_bar': log}


class FedAvgNodePredictor(FedNodePredictor):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, True, *args, **kwargs)

    def aggregate_local_train_state_dicts(self, local_train_state_dicts): # { }字典：state_dict那几项聚合
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:
            agg_state_dict[k] = 0
            for ltsd in local_train_state_dicts:
                agg_state_dict[k] += ltsd[k]
            agg_state_dict[k] /= len(local_train_state_dicts)
        return agg_state_dict


class NoFedNodePredictor(FedNodePredictor):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, False, *args, **kwargs)

    def aggregate_local_train_state_dicts(self, local_train_state_dicts): # [{},{},{}...{} ]列表，207个字典
        return local_train_state_dicts