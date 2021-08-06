## GMPQ: Generalizable Mixed-Precision Quantization via Attribution Rank Preservation

This is the pytorch implementation for the paper: *Generalizable Mixed-Precision Quantization via Attribution Rank Preservation*, which is accepted to ICCV2021. This repo contains searching the quantization policy via attribution preservation on small datasets 
including CIFAR-10, Cars, Flowers, Aircraft, Pets and Food, and finetuning on largescale dataset like ImageNet using our proposed GMPQ.


## Quick Start

### Prerequisites

- python>=3.5
- pytorch>=1.1.0
- torchvision>=0.3.0 
- other packages like numpy and sklearn

### Dataset 

If you already have the ImageNet dataset for pytorch, you could create a link to data folder and use it:
```
# prepare dataset, change the path to your own
ln -s /path/to/imagenet/ data/
```
If you don't have the ImageNet, you can use the following script to download it: 
[https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

For small datasets which we search the quantization policy on, please follow the official instruction:

- [CIFAR-10]()
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

### Searching the mixed-precision quantization policy 
For a specific small dataset, you should first pretrain a full-precision model to provide supervision for attribution rank consistency preservation and save it to `pretrain_model.pth.tar`.

After that, you can start searching the quantization policy. Take ResNet18 and CIFAR-10 for example:

```
CUDA_VISIBLE_DEVICES=0,1 python search_attention.py \
-a mixres18_w2346a2346  -fa qresnet18_cifar  --epochs 25  --pretrained pretrain_model.pth.tar --aw 40 \
--dataname cifar10 --expname cifar10_resnet18  --cd 0.0003   --step-epoch 10    \
--batch-size 256   --lr 0.1   --lra 0.01 -j 16  \
  path/to/cifar10 \
```
 It also supports other network architectures like ResNet50 and other small datasets like Cars, Flowers, Aircraft, Pets and Food.
 
 ### Finetuning on ImageNet

After searching, you can get the optimal quantization policy, with the checkpoint `arch_checkpoint.pth.tar`. You can run the following command to finetune and evaluate the performance on ImageNet dataset. 

```

CUDA_VISIBLE_DEVICES=0,1 python main.py     \
 -a qresnet18                 \
 --ac arch_checkpoint.pth.tar \
 -c checkpoints/train_resnet18   \
 --data_name imagenet          \
 --data path/to/imagenet           \
 --epochs 100                     \
 --pretrained pretrained.pth.tar
 --lr 0.01                    \
 --gpu_id 1,2,3     \
 --train_batch_per_gpu 192              \
 --wd 4e-5                       \
 --workers 32                    \
```