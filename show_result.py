import argparse
import os
from glob import glob

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='result/')
args = parser.parse_args()
file_paths = [y for x in os.walk(args.input_path) for y in glob(os.path.join(x[0], '*.pt'))]
headers = ['1st Task', '2nd Task', '3rd Task']
for file_path in file_paths:
    if 'result.pt' in file_path:
        name = file_path.split('/')[-1].split('.')[0]
        test_accuracies = torch.load(file_path)['test_accuracies']
        for test_results in test_accuracies:
            if type(test_results) is tuple or type(test_results) is list:
                results = zip(headers[:len(test_results)], ['%0.2f%%' % (float(x) * 100) for x in test_results])
            else:
                results = [(headers[0], ': %0.2f%%' % (float(test_results) * 100))]
            print(name, dict(results))
