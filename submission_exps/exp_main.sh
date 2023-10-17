# METR-LA

# GRU (centralized, 63K)
nohup python main.py --dataset METR-LA --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1  --use_curriculum_learning --max_epochs 10 > log/NodePredictor.log  2>&1 &

# GRU (centralized, 727K)
python main.py --dataset METR-LA --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --gpus 0, --use_curriculum_learning

# need --restore_train_ckpt_path
nohup python main.py --dataset METR-LA --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1  --use_curriculum_learning --max_epochs 13 --restore_train_ckpt_path='tb_logs/NodePredictor/version_0/checkpoints/epoch=9-step=10.ckpt'> log/NodePredictor.log  2>&1 &

# GRU (local, 63K)
nohup python main.py --dataset METR-LA --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 1 --max_epochs 1 --sync_every_n_epoch 1 > log/NoFedNodePredictor.log  2>&1 &

# GRU (local, 727K)
python main.py --dataset METR-LA --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 6 --max_epochs 1

# GRU (63K) + FedAvg
nohup python main.py --dataset METR-LA --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 1 --max_epochs 1 --sync_every_n_epoch 1 --cl_decay_steps 1 > log/FedAvgNodePredictor.log  2>&1 &

# GRU (727K) + FedAvg
python main.py --dataset METR-LA --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 8 --sync_every_n_epoch 1 --cl_decay_steps 1

# GRU (63K) + FMTL
nohup python main.py --dataset METR-LA --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 1 --max_epochs 1 --mtl_lambda 0.01 --sync_every_n_epoch 1 > log/FixedGraphFedMTLNodePredictor.log 2>&1 &

# GRU (727K) + FMTL
python main.py --dataset METR-LA --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 8 --mtl_lambda 0.01 --sync_every_n_epoch 5

# CNFGNN (64K + 1M)
nohup python main.py --dataset METR-LA --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 128 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 1 --gcn_on_server  --early_stop_patience 20 --gru_num_layers 1 --max_epochs 1 > log/SplitFedAvgNodePredictor.log 2>&1 &

nohup python main.py --dataset METR-LA --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 128 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --gcn_on_server  --early_stop_patience 20 --gru_num_layers 1 --max_epochs 2 --clusters 80 --server_epoch 1 > log/SplitFedAvgNodePredictor.log 2>&1 &

# PEMS-BAY

# GRU (centralized, 63K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 64 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning

# GRU (centralized, 727K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 64 --hidden_size 200 --gru_num_layers 2 --gpus 0, --use_curriculum_learning

# GRU (local, 63K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 14 --max_epochs 100 --early_stop_patience 20

# GRU (local, 727K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 14 --max_epochs 100 --early_stop_patience 20

# GRU (63K) + FedAvg
python main.py --dataset PEMS-BAY --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 12 --sync_every_n_epoch 1 --cl_decay_steps 1

# GRU (727K) + FedAvg
python main.py --dataset PEMS-BAY --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 12 --sync_every_n_epoch 1 --cl_decay_steps 1

# GRU (63K) + FMTL
python main.py --dataset PEMS-BAY --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 8 --mtl_lambda 0.01 --sync_every_n_epoch 5

# GRU (727K) + FMTL
python main.py --dataset PEMS-BAY --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 8 --mtl_lambda 0.01 --sync_every_n_epoch 5

# CNFGNN (64K + 1M)
python main.py --dataset PEMS-BAY --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 6 --sync_every_n_epoch 1 --server_epoch 20 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 20 --gru_num_layers 1