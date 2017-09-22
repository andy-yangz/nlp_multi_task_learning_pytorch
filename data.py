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
    def __init__(self, path, lang_name):
        
        self.lang_name = lang_name
        # Create Dictionaries
        self.word_dict = Dictionary('Word')
        self.pos_dict = Dictionary('POS')
        self.dep_dict = Dictionary('Dependency')

        self.word_train, self.pos_train, self.dep_train = self.tokenize(os.path.join(path, 'train.conllu'))
        self.word_valid, self.pos_valid, self.dep_valid = self.tokenize(os.path.join(path, 'dev.conllu'))
        self.word_test, self.pos_test, self.dep_test = self.tokenize(os.path.join(path, 'test.conllu'))
        
    def tokenize(self, path):
        "Tokenizes text data file"
        sentences_word_ids = []
        sentences_pos_ids = []
        sentences_dep_ids = []
        sentence_word = []
        sentence_pos = []
        sentence_dep = []
        assert os.path.exists(path)
        # Build the dictionaries from corpus
        print('Loadind Data...')
        with open(path, 'r') as f:
            for line in f:
                # Ignore comments lines
                if line[0] == '#':
                    continue
                row = line.strip().split()
                if len(row) != 0:
                    print(row)
                    sentence_word.append(row[1])
                    sentence_pos.append(row[3])
                    sentence_dep.append((row[6], row[7]))
                    self.word_dict.add_word(row[1])
                    self.pos_dict.add_word(row[3])
                    self.dep_dict.add_word(row[7])
                else: 
                    # Encode sentence
                    sentence_word_id = torch.LongTensor([self.word_dict.word2idx[word] for word in sentence_word])
                    sentence_pos_id = torch.LongTensor([self.pos_dict.word2idx[pos] for pos in sentence_pos])
                    sentence_dep_id = torch.LongTensor([self.dep_dict.word2idx[dep[1]] for dep in sentence_dep])
                    # Append sentence
                    sentences_word_ids.append(sentence_word_id)
                    sentences_pos_ids.append(sentence_pos_id)
                    sentences_dep_ids.append(sentence_dep_id)
                    sentence_word = []
                    sentence_pos = []
                    sentence_dep = []                    

        return sentences_word_ids, sentences_pos_ids, sentences_dep_ids
