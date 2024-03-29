#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 3 \
    --hidden-dim 64 \
    --epochs 10 \
    --weight-decay 0.0001 \
    --momentum 0.0 \
    --batch-size 64 \
    --lr 0.001 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
