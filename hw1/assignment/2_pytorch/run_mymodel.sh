#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 1 \
    --hidden-dim 10 \
    --epochs 500 \
    --weight-decay 0.01 \
    --momentum 0.0 \
    --batch-size 50 \
    --lr 0.1 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
