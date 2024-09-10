#from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
dataset = load_dataset("text", data_files={"train": ["fly_neg.txt"]})


def add_label(example):
    example['label'] = 'negative'
    return example

dataset = dataset['train'].select(range(50000))
dataset = dataset.map(add_label)
dataset.save_to_disk('neg_species2')
print(dataset)
