# NLP Multi-task Learning using PyTorch
This repo adapts the code found [here](https://github.com/andy-yangz/nlp_multi_task_learning_pytorch/) to support NER as an additional task, as well as pre-trained embeddings.

The two tasks we use here is POS Tagging and Chunking, with a branch to do NER as well.

As below, we build the model, normally several layers of RNN above embedding. Later, we train specific task on different layer according to its complexity. For example, we think Chunking is a higher task than POS Tagging, inspired by *[Deep multi-task learning with low level tasks supervised at lower layers](http://anthology.aclweb.org/P16-2038)*.

![img](https://ws3.sinaimg.cn/large/006tNbRwgy1fuchyzqmynj30ik0aogm6.jpg)

## Install

Requires PyTorch 0.3.1 as well as torchtext. Follow install below (assumes you have a conda environment):
```bash
conda install pytorch=0.3.1 -c pytorch
pip install torchtext==0.2.3
```
We highly suggest using a CUDA based machine.

## Data

We use the conll2003 data found [here](https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en). Follow the below instructions to format the data correctly:

```bash
cd nlp_multi_task_learning_pytorch/data/
mkdir pos_chunk_ner/
wget https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en//test.txt -P pos_chunk_ner/
wget https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/valid.txt -P pos_chunk_ner/
wget https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/train.txt -P pos_chunk_ner/
python format_data.py --input_path pos_chunk_ner/ --output_path pos_chunk/
python format_data.py --input_path pos_chunk_ner/ --output_path pos_ner/ --ner
```

## Usage
```bash
mkdir result/
bash run.sh data/pos_chunk/ # Swap run.sh to run_bidirectional.sh to benchmark bidirectional models
```

This will create a folder called result/ containing all the pth files containing the models/results. Change data/pos_chunk to data/pos_ner if you want to record POS + NER results.

Run show_result.py to see the test results per model in this directory.

### Using NER

Also supported is adding NER as an additional sub-task. This can be done by swapping to the `NER` branch and running the below command.

```bash
mkdir result/
bash run.sh data/pos_chunk_ner/
```

### Acknowledgements

This repo adapts the code found [here](https://github.com/andy-yangz/nlp_multi_task_learning_pytorch/) by Andy who we'd like to thank!
