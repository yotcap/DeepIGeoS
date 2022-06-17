# DeepIGeoS
Implementation the DeepIGeoS Paper

### Environments
`Python 3.7.11`
```
dotmap
GeodisTK
opencv-python
tensorboard
torch
torchio
torchvision
tqdm
PyQt5
```

## Datasets
执行 `load_datasets.sh` 加载数据集：[BraTS 2021](https://arxiv.org/pdf/2107.02314.pdf)
```
$ bash load_datasets.sh
```

## Train

### P-Net
```
$ python train_pnet.py -c configs/config_pnet.json
```

### R-Net
```
$ python train_rnet.py -c configs/config_rnet.json
```

### Tensorboard
```
$ tensorboard --logdir experiments/logs/
```

## Run

### Simple QT Application
基于 QT 创建用于鼠标点击的交互行为应用，只需要运行主代码即可：
```
$ python main_deepigeos.py
```
