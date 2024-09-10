from GENA_LM.src.gena_lm.modeling_bert import BertForSequenceClassification
#from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import sklearn
from datasets import load_dataset, load_metric,load_from_disk
import evaluate

from transformers import AutoTokenizer, AutoModel,TrainingArguments,Trainer

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base', trust_remote_code=True)

# Step 1: Import the tokenizer and the model
#tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
#model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base')

# Step 2: Load your dataset (example with a Hugging Face dataset)
dataset = load_from_disk('species2')#rna_new/rna_trim') #rna_trimrna_trimload_dataset('text', data_files=['rna_new/pos.txt','rna_new/neg.txt'])
#dataset = dataset['train'].train_test_split(test_size=0.10) #05)

 #load_dataset('csv', data_files={'train':"healthdata/train1.csv",'test':"healthdata/test.csv"}) #"species/train1.csv",'test':"species/test.csv"}) # "dnabert2_lnc/train2.csv",'test':"dnabert2_lnc/test.csv"}) #"prom_nonpromdata/train1.csv",'test':"prom_nonpromdata/test.csv"}) #species/train.csv",'test': "species/test.csv"}) #pecies_hf")
#dataset = load_from_disk('rna_v2')

#dataset = load_from_disk('rna_v2')
print(dataset)
# Step 3: Preprocess the data
def tokenize_function(examples):
    print(examples)
    return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=20)
'''

train_dataset = dataset['train'].map(tokenize_function, batched=True)
test_dataset = dataset['test'].map(tokenize_function, batched=True)
# Step 4: Prepare for training
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
'''

def compute_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")
    metric5 = evaluate.load("matthews_correlation")
   # metric5 = load_metric('matthews_correlation')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels, average="micro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="micro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="micro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]
    mcc = metric5.compute(predictions=predictions, references=labels)#["matthews_correlation"]


    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,"mcc":mcc}



def compute_metrics_(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    # Load metrics
    f1_metric = load_metric('f1')
    mcc_metric = load_metric('matthews_correlation')

    # Compute F1 and MCC
    f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')
    mcc = mcc_metric.compute(predictions=preds, references=labels)

    return {"f1": f1['f1'], "mcc": mcc['matthews_correlation']}

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=3,
)
print(training_args)
# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset =test_dataset,
    compute_metrics=compute_metrics

)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model
model.save_pretrained("./species2")
tokenizer.save_pretrained("./fine-tuned-gena-species2")
eval_results = trainer.evaluate()
print(eval_results)
