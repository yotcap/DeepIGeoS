import torch
import random
import numpy as np
import torch.distributed as dist

import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

from utils.utils import get_args
from utils.dirs import create_dirs
from utils.device import device_config
from utils.logger import MetricsLogger
from utils.config import process_config

from models.networks import build_model
from data_loaders.brats2021_3d import get_dataloaders
from trainers.brats_3d_rnet_trainer import Brats3dRnetTrainer


def main():
    # 获取配置路径，处理配置文件
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # 创建实验的 dir
    create_dirs((
        config.exp.tensorboard_dir, 
        config.exp.last_ckpt_dir, 
        config.exp.best_ckpt_dir,
        config.exp.val_pred_dir
    ))

    # 硬件配置（GPU / CPU）
    device_config(config)

    # 创建 logger
    if config.exp.multi_gpu:
        # 进程之间共享
        BaseManager.register('MetricsLogger', MetricsLogger)
        manager = BaseManager()
        manager.start()
        logger = manager.MetricsLogger(config)

    # 运行主 workers
    if config.exp.multi_gpu:
        mp.spawn(
            main_worker, 
            nprocs=config.exp.world_size, 
            args=(config, logger,)
        )
    else:
        # 随机种子
        random.seed(1111)
        np.random.seed(1111)
        torch.manual_seed(1111)

        # 载入数据集
        dataloaders = get_dataloaders(config)
        
        # 构建 model
        model = build_model(config)

        # 创建 logger
        logger = MetricsLogger(config)

        # 创建 trainer
        trainer = Brats3dRnetTrainer(model, dataloaders, config, logger)

        # 训练
        trainer.train()


def main_worker(rank, config, logger):
    # 初始化 worker group
    setup(rank, config.exp.world_size)

    # 设置 cuda visible device
    torch.cuda.set_device(rank)
    
    # 为每个 GPU 设置
    config.data.num_workers = int(config.data.num_workers / config.exp.ngpus_per_node)
    config.exp.device = torch.device(f"cuda:{rank}")
    config.exp.rank = rank

    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)
    
    # 载入数据集
    dataloaders = get_dataloaders(config)
    
    # 构建模型
    model = build_model(config)

    # 创建 trainer
    trainer = Brats3dRnetTrainer(model, dataloaders, config, logger)

    # 训练
    trainer.train()

    # 清除进程
    cleanup()


def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=rank
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()