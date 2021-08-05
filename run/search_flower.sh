export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore search_attention.py  \
 -a mixres18_w234a234   \
 --epochs 25    \
 --dataname flower \
 --expname flower  \
 --step-epoch 10    \
 --batch-size 8   \
 --lr 0.1   \
 --lra 0.01 \
 --cd 0.00335   \
 -j 16  \
 /home/wzw/SSD/flower_data \