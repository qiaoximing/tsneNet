# tsneNet

Transferring neural network knowledge with t-sne regularizer.

## Overview

Deep neural networks have great performance in many domains, however their computation costs are not always affordable in resource limited environments. One solution is to shrink networks' sizes, but that usually leads to huge reduction in their capabilities. Knowledge transfer is an attractive idea to lower the reduction, but there are few reports about its successful applying on neural networks. In this paper, we borrow the idea of t-SNE algorithm, a popular visualization method that can preserve local structures of high dimension data, to solve the problem of applying knowledge transfer in neural networks. We design a novel t-SNE regularizer, which allows a student network to learn the training data structure from a teacher network, and improve in accuracy. The training data structure is represented in high dimension feature spaces. We test the proposed method on different scales of datasets. By using pre-trained teacher networks, the error rate of student networks are reduced from 2.4\% to 1.7\% on MNIST, and from 62\% to 49\% on a 10-label subset of ImageNet. These results show that the t-SNE regularizer is effective in improving neural networks' accuracy, and is highly scalable in multiple datasets and different networks.

Full paper on https://github.com/qiaoximing/qiaoximing.github.io/blob/master/v3-Knowledge%20Transfer%20in%20Neural%20Networks%20with%20t-SNE%20Regularizer.pdf

## Run the code

1. Install tensorflow on python3
2. `python3 tsneNet.py`

## Thanks

`tsne3.py` is ported from https://lvdmaaten.github.io/tsne/code/tsne_python.zip, adjusted `print()` style to fit python3.
