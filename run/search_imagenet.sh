export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python -W ignore search.py  \
 -a mixres18_w1234a234   \
 --epochs 25    \
 --step-epoch 10    \
 --batch-size 512   \
 --lr 0.1   \
 --lra 0.01 \
 --cd 0.00335   \
 -j 16  \
 /home/wzw/SSD/ILSVRC2012   \