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
        self.token_embedding_table = nn.Embedding(tokenizer.n_vocab, 100)
        self.linear_layer = nn.Linear(100, tokenizer.n_vocab)
        
    def forward(self, x, targets=None):
        # What we send to miodle is 4 sentences of 8 words each.What we expect from the model is to predict the next word in the sentence.In one pass all the 4 sentences in the batch is processes simultaneously by looking up the token in the embedding table.
        #Each token in the embedding table is of 64 dimensions.
        print(self.token_embedding_table)
        embedded_inp = self.token_embedding_table(x)
        print(f'When we lookup embedding table with the input data we get embedded input with shape {embedded_inp.shape}')
        logits = self.linear_layer(embedded_inp)
        print(f'When we apply a matrix multiplication on a table with 4 * 8 x 100 with 100 x 1000k we get an output of shape {logits.shape}')
        if targets is None:
            loss = None
        else:   
            B, T, C = logits.shape
            print(logits)
            logits = logits.view(B*T, C)
            print(f' logits after reshape {logits}')
            # targets = torch.ones(32, dtype=int)*5
            targets = targets.view(B*T)
            print(targets)
            # What logits are giving is some scores for 100 k tokens that should appear at this position and targets gas the actuak word at that position
            # Cross entropy will apply softmax whihc is it will exponentiate all the 100k scores and divide by the sum.This will convert the scores into probs and the word with the highest probabilty becomes the predicted word at that position and then we just compare with the actual target and calculate the loss.
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