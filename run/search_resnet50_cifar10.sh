export CUDA_VISIBLE_DEVICES=0,1,2
python -W ignore search_attention.py  \
 -a mixres50_w1234a234  \
 -fa qresnet50  \
 --epochs 25    \
 --dataname cifar10 \
 --expname resnet50_cifar10_attention_0.00001  \
 --cd 0.00001   \
 --resume /home/wzw/xh/MIPS/resnet50_cifar10_attention_0.00001/arch_model_best.pth.tar  \
 --step-epoch 10    \
 --batch-size 96   \
 --lr 0.1   \
 --lra 0.01 \
 -j 16  \
 ~/SSD/cifar10_data/ \