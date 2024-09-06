import pandas as pd
import tiktoken
import torch as torch

# Load the data
data = pd.read_csv('./data/corona_nlp_train.csv', encoding='utf-8', encoding_errors='ignore')
#print(data.head())

text = data['OriginalTweet']
#print(text.head())

# Make it one big text

text = ' '.join(text)

#print(text[:1000])

chars = sorted(list(set(text)))
#print(chars)

#Lets encode the text using tiktokenizer

tokenizer = tiktoken.get_encoding('cl100k_base')
encoded_text = tokenizer.encode(text) # This will return a list of integers
#print(encoded_text[:100])

# Lets now split the data

n = int(len(encoded_text) * 0.9)
train_data = encoded_text[:n]
val_data = encoded_text[n:]

block_size = 8
#print(train_data[:block_size +1])

x = train_data[:block_size]
y = train_data[1:block_size + 1]
batch_size = 4


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    #x = torch.stack([data[i:i+block_size] for i in ix][0])
    x = torch.stack(torch.tensor)(0)
    print(ix)

get_batch('train')