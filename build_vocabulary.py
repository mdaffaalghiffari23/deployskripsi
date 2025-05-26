#!/usr/bin/env python3
"""
Create vocabulary file matching the training process
"""

import pandas as pd
import nltk
import itertools
import re
from nltk.tokenize import word_tokenize
import pickle
import json

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def build_vocabulary_like_training(csv_path):
    """Build vocabulary exactly like in training process"""
    
    # Load CSV
    df = pd.read_csv(csv_path, usecols=['id_code', 'caption'])
    
    # Create combined data like in training
    img_id = df['id_code'].values.tolist()
    img_caption = df['caption'].values.tolist()
    
    combined_data = []
    for i in range(len(img_id)):
        if i < len(img_caption):
            combined_data.append(f"{img_id[i]}\t{img_caption[i]}")
    
    # Process captions exactly like in training
    imgcaption_dict = {}
    for i in range(0, len(combined_data), 1):
        cap = combined_data[i].strip()
        cap = cap.split('\t')
        imgcaption_dict[cap[0]] = re.sub(r'[^a-zA-Z0-9\s]', '', cap[1]).strip().lower().replace('  ', ' ')
    
    # Tokenize captions
    imgcaption_token = []
    imgcaption_wordtoken = []
    for (i, j) in imgcaption_dict.items():
        imgcaption_token.append([i, nltk.word_tokenize(j)])
        imgcaption_wordtoken.append(nltk.word_tokenize(j))
    
    # Create vocabulary exactly like training
    alltokens = itertools.chain.from_iterable(imgcaption_wordtoken)
    wordtoid = {token: idx for idx, token in enumerate(set(alltokens))}
    
    alltokens = itertools.chain.from_iterable(imgcaption_wordtoken)
    idtoword = [token for idx, token in enumerate(set(alltokens))]
    
    # Convert tokens to IDs
    imgcaption_token_id = [[wordtoid.get(token, -4) + 4 for token in x[1]] for x in imgcaption_token]
    
    # Add special tokens
    wordtoid['<unknown>'] = 0
    wordtoid['<start>'] = 1
    wordtoid['<end>'] = 2
    wordtoid['<pad>'] = 3
    
    # Adjust token IDs
    for (_, i) in wordtoid.items():
        if i < 0:  # Only adjust the special tokens we just added
            wordtoid[_] = i + 4
    
    # Create ID to word mapping
    idtoword_dict = {}
    cnt = 4
    for i in idtoword:
        idtoword_dict[cnt] = i
        cnt += 1
    idtoword_dict[0] = '<unknown>'
    idtoword_dict[1] = '<start>'
    idtoword_dict[2] = '<end>'
    idtoword_dict[3] = '<pad>'
    
    return wordtoid, idtoword_dict

def main():
    print("Building vocabulary from train.csv...")
    
    wordtoid, idtoword_dict = build_vocabulary_like_training("train.csv")
    
    print(f"Vocabulary size: {len(wordtoid)}")
    print(f"WordtoID keys: {list(wordtoid.keys())}")
    print(f"IDtoWord keys: {list(idtoword_dict.keys())}")
    
    # Save vocabulary
    vocab_data = {
        'wordtoid': wordtoid,
        'idtoword': idtoword_dict
    }
    
    with open('vocabulary.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print("Vocabulary saved to vocabulary.json")
    
    # Also save as pickle for faster loading
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab_data, f)
    
    print("Vocabulary saved to vocabulary.pkl")

if __name__ == "__main__":
    main()
