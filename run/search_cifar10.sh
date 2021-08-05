export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore search_attention.py  \
 -a mixres18_w234a234  \
 -fa qresnet18  \
 --epochs 25    \
 --dataname cifar10 \
 --expname cifar10_0.0003  \
 --cd 0.0003   \
 --step-epoch 10    \
 --batch-size 256   \
 --lr 0.1   \
 --lra 0.01 \
 -j 16  \
 /home/wzw/SSD/cifar10_data \