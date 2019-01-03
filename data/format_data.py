import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='pos_chunk_ner/')
parser.add_argument('--output_path', default='output/')
parser.add_argument('--ner', action='store_true', help='Replace chunk label with NER label')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
for file_name in os.listdir(args.input_path):
    input_path = os.path.join(args.input_path, file_name)
    with open(input_path) as f:
        lines = f.read().strip().split('\n')[1:]  # Remove docstart
        lines = [x.strip() for x in lines if x.strip()]  # Remove empty lines
        lines = [x.split(' ') for x in lines]
        if args.ner:
            lines = [' '.join((x[0], x[1], x[3])) for x in lines]  # Remove chunk label
        else:
            lines = [' '.join((x[0], x[1], x[2])) for x in lines]  # Remove ner label
    output_path = os.path.join(args.output_path, file_name)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
