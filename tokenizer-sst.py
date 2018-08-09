from __future__ import print_function
import argparse
import json
import random
from util import Dictionary
import spacy

# python tokenizer-sst.py --input ./data/SST-2/sst_small.tsv --output ./data/SST-2/tokenized-sst-small.json --dict ./data/SST-2/sst-small.dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Tokenizer')
    parser.add_argument('--input', type=str, default='', help='input file')
    parser.add_argument('--output', type=str, default='', help='output file')
    parser.add_argument('--dict', type=str, default='', help='dictionary file')
    args = parser.parse_args()
    
    tokenizer = spacy.load('en')
    dictionary = Dictionary()
    dictionary.add_word('<pad>')  # add padding word

    with open(args.output, 'w') as fout:
        lines = open(args.input).readlines()
        # random.shuffle(lines)
        for i, line in enumerate(lines):

            each = line.split('\t')
            text = each[0].strip()
            label = each[1].strip()

            words = tokenizer(' '.join(text.split()))
            data = {
                'label': int(label),
                'text': list(map(lambda x: x.text.lower(), words))
            }
            fout.write(json.dumps(data) + '\n')
            for item in data['text']:
                dictionary.add_word(item)
            if i % 100 == 99:
                print('%d/%d files done, dictionary size: %d' %
                      (i + 1, len(lines), len(dictionary)))

            if i > 10000:
                break

        fout.close()

    with open(args.dict, 'w') as fout:  # save dictionary for fast next process
        fout.write(json.dumps(dictionary.idx2word) + '\n')
        fout.close()