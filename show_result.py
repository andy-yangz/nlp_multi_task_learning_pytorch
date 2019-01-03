import os
from glob import glob

import torch

path = './result'
file_paths = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.pt'))]

results = []
for file_path in file_paths:
    if 'result.pt' in file_path:
        name = file_path.split('/')[-1].split('.')[0]
        best_accuracy = torch.load(file_path)['test_accuracies']
        for val in best_accuracy:
            if type(val) is tuple or type(val) is list:
                print(name, ['%0.2f%%' % (float(x) * 100) for x in val])
            else:
                print(name, ['%0.2f%%' % (float(val) * 100)])