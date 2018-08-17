# nlp_multi_task_learning_pytorch
A basic multitask learning architecture for Natural Language Processing of Pytorch implementation.

The two tasks we use here is POS Tagging and Chunking. 

As below, we build the model, normally several layers of RNN above embedding. Later, we train specific task on different layer according to its complexity. For example, we think Chunking is a higher task than POS Tagging, inspired by *[Deep multi-task learning with low level tasks supervised at lower layers](http://anthology.aclweb.org/P16-2038)*.

![img](https://ws3.sinaimg.cn/large/006tNbRwgy1fuchyzqmynj30ik0aogm6.jpg)

## Running Examples

You can check several running examples in `run.sh`. I will explain one here.

```
echo "Joint Training on the different level"
python main.py --data './data' \  # The directory put the training data
        --emsize 256 \	# embedding size
        --npos_layers 1 \	# number of POS tagging training layer
        --nchunk_layers 2 \	# number of chunking training layer
        --nhid 128 \	# number of hidden states for RNN
        --batch_size 128 \
        --seq_len 10 \	# sequence length
        --cuda \	# enbale GPU
        --train_mode 'Joint' \
        --epochs 300 \	
        --log_interval 20 \
        --save './result/joint_diff'
```

## Current Done

  - A basic architecture for POS tagging and chunking
  - Explorate the best hyperparameters of NN

## Todo
  - [ ] Chunking exploration
