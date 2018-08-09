import numpy as np
# from __future__ import print_function
# import argparse
import json
import random
# from util import Dictionary
# import spacy

RATIO_TRAINING = .6
RATIO_VALIDATION = .2
RATIO_TESTING = .2

# total_data_length = len(data)
total_data_length = 1000

if __name__ == '__main__':

    # input_file = './data/yelp_academic_dataset_review.json'
    # input_file = './data/tokenized-yelp.json'
    input_file = './data/SST-2/tokenized-sst-10k.json'

    with open(input_file, 'r') as fin:
        data = fin.readlines()
        random.shuffle(data)

        tr_size = int(total_data_length*RATIO_TRAINING)
        va_size = int(total_data_length*RATIO_VALIDATION)
        te_size = int(total_data_length*RATIO_TESTING)

        train_data, valid_data, test_data = data[:tr_size], data[tr_size:tr_size+va_size], data[tr_size+va_size:tr_size+va_size+te_size]
        
        print(len(train_data))
        print(len(valid_data))
        print(len(test_data))

        output_file_train = './data/SST-2/tokenized-sst-10k-train.json'
        output_file_valid = './data/SST-2/tokenized-sst-10k-val.json'
        output_file_test = './data/SST-2/tokenized-sst-10k-test.json'

        with open(output_file_train, 'w') as fout_train:
            for i, line in enumerate(train_data):
                item = json.loads(line)
                fout_train.write(json.dumps(item) + '\n')

            fout_train.close()

        with open(output_file_valid, 'w') as fout_valid:
            for i, line in enumerate(valid_data):
                item = json.loads(line)
                fout_valid.write(json.dumps(item) + '\n')

            fout_valid.close()

        with open(output_file_test, 'w') as fout_test:
            for i, line in enumerate(test_data):
                item = json.loads(line)
                fout_test.write(json.dumps(item) + '\n')

            fout_test.close()
        
        fin.close()