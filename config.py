arg_default = {
	'emsize': int(300),
	'nhid': int(300),
	'nlayers': int(2),
	'attention-unit': int(350),
	'attention-hops': int(1),
	'dropout': float(0.5),
	'clip': float(0.5),
	'nfc': int(512),
	'lr': float(0.001),
	'epochs': int(40),
	'seed': 1111,
	'cuda': bool(True),
	'log-interval': int(200),
	'save': '',
	'dictionary': '',
	'word-vector': '',
	'train-data': '',
	'val-data': '',
	'test-data': '',
	'batch-size': int(32),
	'class-number': int(2),
	'optimizer': 'Adam',
	'penalization-coeff': float(1.0)
}

# SST Task - original
arg_sst = arg_default
arg_sst['save'] = 'models/model-sst-1000.pt'
arg_sst['dictionary'] = 'data/SST-2/sst.dict'
arg_sst['word-vector'] = 'data/GloVe/glove.42B.300d.pt'
arg_sst['train-data'] = 'data/SST-2/tokenized-sst-1000-train.json'
arg_sst['val-data'] = 'data/SST-2/tokenized-sst-1000-val.json'
arg_sst['test-data'] = 'data/SST-2/tokenized-sst-1000-test.json'
arg_sst['epochs'] = int(10)

# SST Task - small 10k
arg_sst10k = arg_default
arg_sst10k['save'] = 'models/model-sst-10k.pt'
arg_sst10k['dictionary'] = 'data/SST-2/sst-10k.dict'
arg_sst10k['word-vector'] = 'data/GloVe/glove.42B.300d.pt'
arg_sst10k['train-data'] = 'data/SST-2/tokenized-sst-10k-train.json'
arg_sst10k['val-data'] = 'data/SST-2/tokenized-sst-10k-val.json'
arg_sst10k['test-data'] = 'data/SST-2/tokenized-sst-10k-test.json'
arg_sst10k['epochs'] = int(1)
