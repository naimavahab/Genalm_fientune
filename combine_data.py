from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk,concatenate_datasets

posdataset = load_from_disk("pos_species2")
negdataset = load_from_disk("neg_species2")
posdataset =posdataset.select(range(50000))
print(posdataset)
print(negdataset)
dataset_cc = concatenate_datasets([posdataset, negdataset])
dataset_cc = dataset_cc.class_encode_column("label")

dataset_cc = dataset_cc.shuffle(seed=42)
dataset_cc = dataset_cc.train_test_split(test_size=0.10)
dataset_cc.save_to_disk('species2')
print(dataset_cc)
print(dataset_cc['train'][500])
print(dataset_cc['test'][0])
print(len(dataset_cc['train']))
print(len(dataset_cc['test']))

