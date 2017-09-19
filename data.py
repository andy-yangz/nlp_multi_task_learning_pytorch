import os
import torch 


class Dictionary(object):
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.nwords
            self.idx2word.append(word)
            self.nwords += 1

    def __str__(self):
        return "%s dictionary has %d kinds of tokens." \
                % (self.name, self.nwords)


class Corpus(object):
    def __init__(self, path):
        self.word_dict = Dictionary('Word')
        self.pos_dict = Dictionary('POS')
        self.chunk_dict = Dictionary('Chunk')

        self.word_train, self.pos_train, self.chunk_train = self.tokenize(os.path.join(path, 'train.txt'))
        self.word_valid, self.pos_valid, self.chunk_valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.word_test, self.pos_test, self.chunk_test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        "Tokenizes text data file"
        assert os.path.exists(path)
        # Build the dictionaries from corpus
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                try: 
                    word, pos, chunk = line.strip().split()
                except: 
                    continue
                tokens += 1
                self.word_dict.add_word(word)
                self.pos_dict.add_word(pos)
                self.chunk_dict.add_word(chunk)

        with open(path, 'r') as f:
            word_ids = torch.LongTensor(tokens)
            pos_ids = torch.LongTensor(tokens)
            chunk_ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                try: 
                    word, pos, chunk = line.strip().split()
                except: 
                    continue
                word_ids[token] = self.word_dict.word2idx[word]
                pos_ids[token] = self.pos_dict.word2idx[pos]
                chunk_ids[token] = self.chunk_dict.word2idx[chunk]
                token += 1

        return word_ids, pos_ids, chunk_ids
