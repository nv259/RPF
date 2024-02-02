from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

import re

import pandas as pd
import h5py
import joblib


def load_collections(attr_ids):
    if type(attr_ids) is not list:
        attr_ids = [attr_ids]
    
    collections_id = []
    collections = []
     
    for attr_id in attr_ids:
        collection_id = joblib.load("collections/c_idxs_" + str(attr_id) + ".joblib")
        collection = joblib.load("collections/c_feats_" + str(attr_id) + ".joblib")
        collections_id.append(collection_id)
        collections.append(collection)
         
    return collections_id, collections


def collections_to_h5py(n_attrs=8):
    collections_id, collections = load_collections([0, 1, 2, 3, 4, 5, 6, 7])
    
    with h5py.File("collections/data.h5", 'w') as file:
        for i in range(n_attrs): 
            file.create_dataset('attr' + str(i), data=collections[i])
            
# collections_to_h5py()


def split_sentences(text):
    # Extend the pattern to include the specified special characters
    # Escape special characters that have meanings in regex with a backslash
    pattern = r'\s+and\s+|\s+or\s+|\s+with\s+|[+;,\'"-]'
    
    # Use re.split to split the text based on the pattern
    sentences = re.split(pattern, text)
    
    return sentences


def finetune_pretrained(pretrained_path, data_path):
    # TODO: generalize for other dataset beside FashionConversationTwitter
    # 1) Load dataset
    dataset = pd.read_csv(data_path)
    texts = dataset['Caption'].fillna('') + ' ' + dataset['Hashtags'].fillna('')
    
    # 2) Load tokenizer, model, data collator
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    encodings = tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    model = AutoModelForMaskedLM.from_pretrained(pretrained_path)
    
    # 3) Set up training
    training_args = TrainingArguments(
        output_dir = './finetuned-results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        data_collator=data_collator,
        train_dataset=encodings
    )
    
    # 4) Train and save model
    trainer.train()
    model.save_pretrained(pretrained_path)
    
# finetune_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2')