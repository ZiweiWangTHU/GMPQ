export CUDA_VISIBLE_DEVICES=0,1
python -W ignore main.py    \
 -a quantres18_cfg  \
 --epochs 95    \
 --step-epoch 30    \
 -j 16  \
 -b 640  \
 --ac cifar10_attention_0.00001/arch_model_best.pth.tar   \
 ~/ssd/ILSVRC2012