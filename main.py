import os
from argparse import ArgumentParser

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import models
from utils import SaveNodeEncodings
import logging # 这个是通用的python log库，和pytorch无关 


def main(args):
    torch.set_float32_matmul_precision('medium')
    seed_everything(args.seed)  # Global seed set to
    tb_logger = TensorBoardLogger(save_dir="tb_logs/", name = args.model_name + args.base_model_name) # 对不同的model分文件夹存储

    trainer = Trainer(
        default_root_dir='artifacts', # 如果存在logger，就使用logger中的路径
        deterministic=True,
        callbacks=[SaveNodeEncodings()],
        logger = tb_logger,
        max_epochs=args.max_epochs
    )

    if args.train:
        model = getattr(models, args.model_name)(args)
        if args.restore_train_ckpt_path != '':     
            trainer.fit(model, ckpt_path=args.restore_train_ckpt_path)
        else:
            trainer.fit(model)

if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    
    num_gpus = torch.cuda.device_count()
    gpu_list = [str(i) for i in range(num_gpus)]
    gpu_string = ','.join(gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_string
    
    logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='## %Y-%m-%d %H:%M:%S')
    torch.multiprocessing.set_start_method('spawn')

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='NodeClassifier', help='name of the model')
    parser.add_argument('--base_model_name', type=str, default='MLP')
    parser.add_argument('--prop_model_name', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--save_node_encodings_test', action='store_true')
    parser.add_argument('--load_test_ckpt_path', type=str, default='')
    parser.add_argument('--notrain', dest='train', action='store_false')
    parser.add_argument('--restore_train_ckpt_path', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=1)

    temp_args, _ = parser.parse_known_args()
    ModelClass = getattr(models, temp_args.model_name)
    BaseModelClass = getattr(models, temp_args.base_model_name, None)
    PropModelClass = getattr(models, temp_args.prop_model_name, None)
    parser = ModelClass.add_model_specific_args(parser)
    if BaseModelClass is not None:
        parser = BaseModelClass.add_model_specific_args(parser)
    if PropModelClass is not None:
        parser = PropModelClass.add_model_specific_args(parser)

    args = parser.parse_args()
    logging.warning(args)
    main(args)
