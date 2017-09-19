import os
import torch

f = []
path = './result'
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break

results = []
for file in f:
    if 'result' in f:
        name = f.split('/')[-1].split('.')[0]
        best_accuracy = torch.load(f)
        results.append([name, best_accuracy])

print(results)