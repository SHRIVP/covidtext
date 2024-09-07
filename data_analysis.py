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
    x = torch.stack(([torch.tensor(data[i:i+block_size]) for i in ix]))
    # creates the labels such that if the input is i labels is learning.when the input is i learrning the labels is deep etc.
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x, y

xb, yb = get_batch('train')
print(f' the input for this model:  {xb}')
print(f' the output matrix for the model: {yb}')

# Now we can create the model  

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # Emebedding table is a 64 by 64 matrix that will be learned.Each row in the table is a token in the tiktoken vocab which is around 100 k in our case.And each token ia vector in 64 dimensions.Each of the 64 dimensions is learning something about that token.
        self.token_embedding_table = nn.Embedding(tokenizer.n_vocab, vocab_size)
        
    def forward(self, x, targets=None):
        # What we send to miodle is 4 sentences of 8 words each.What we expect from the model is to predict the next word in the sentence.In one pass all the 4 sentences in the batch is processes simultaneously using some kind of weird :) matrix multiplication.The details of how the multplications works is oos for this comment.
        print(self.token_embedding_table)
        logits = self.token_embedding_table(x)
        print(logits)
        if targets is None:
            loss = None
        else:   
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
        return logits
    
m = BigramLanguageModel(64)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)