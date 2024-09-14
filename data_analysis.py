import pandas as pd
import tiktoken
import torch as torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

# Load the data
data = pd.read_csv('./data/corona_nlp_train.csv', encoding='utf-8', encoding_errors='ignore')
#print(data.head())

text = data['OriginalTweet']
#print(text.head())

# Make it one big text

text = ' '.join(text)


#Lets encode the text using tiktokenizer

tokenizer = tiktoken.get_encoding('cl100k_base')
encoded_text = tokenizer.encode(text) # This will return a list of integers
#print(encoded_text[:100])

# Lets now split the data

n = int(len(encoded_text) * 0.9)
train_data = encoded_text[:n]
val_data = encoded_text[n:]

block_size = 16
#print(train_data[:block_size +1])

x = train_data[:block_size]
y = train_data[1:block_size + 1]
batch_size = 4
n_embd=4
learning_rate = 1e-3
max_iters = 3000
eval_interval = 200


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack(([torch.tensor(data[i:i+block_size]) for i in ix]))
    # print(f' the input for this batch:  {x}')
    # creates the labels such that if the input is i labels is learning.when the input is i learrning the labels is deep etc.
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x, y



@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


# Now we learn Single Head Self Attention.The magic of key,query and value.
# Query is something that I am looking for , key is what I have and if key and query matches then value is what I 
# have to offer.
# Every word in the sentence will have these 3 attributes associated with them
# Attention is a kind of weighted aggregation.What does it mean that while predicting the enxt work in the
# sentence I don't need to give same weightage to all the words that I have seen so far.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
                             
    def forward(self, idx, targets=None):
        B,T,C = idx.shape
        k = self.key(idx)
        q = self.query(idx)
        wei = q @ k.transpose(-2,-1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # I still don't understand why we do softmax.Interview answer to convert scores to probs.why ? don't know
        wei = F.softmax(wei, dim=1)
        v = self.value(idx)
        out = wei @ v
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # Emebedding table is a 64 by 64 matrix that will be learned.Each row in the table is a token in the tiktoken vocab which is around 100 k in our case.And each token ia vector in 64 dimensions.Each of the 64 dimensions is learning something about that token.
        self.token_embedding_table = nn.Embedding(tokenizer.n_vocab, n_embd)
        self.position_embedding_table = nn.Embedding(tokenizer.n_vocab, n_embd)
        self.sa_head = Head(n_embd)
        self.linear_layer = nn.Linear(n_embd, tokenizer.n_vocab)
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        # What we send to miodle is 4 sentences of 8 words each.What we expect from the model is to predict the next word in the sentence.In one pass all the 4 sentences in the batch is processes simultaneously by looking up the token in the embedding table.
        #Each token in the embedding table is of 64 dimensions.
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        # print(f'When we lookup embedding table with the input data we get embedded input with shape {embedded_inp.shape}')
        # apply on head of self attention
        x = self.sa_head(x)
        logits = self.linear_layer(x)
        # print(f'When we apply a matrix multiplication on a table with 4 * 8 x 100 with 100 x 1000k we get an output of shape {logits.shape}')
        if targets is None:
            loss = None
        else:   
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            # print(f' logits after reshape {logits}')
            targets = targets.view(B*T)
            # What logits are giving is some scores for 100 k tokens that should appear at this position and targets has the actuak word at that position
            # Cross entropy will apply softmax whihc is it will exponentiate all the 100k scores and divide by the sum.This will convert the scores into probs and the word with the highest probabilty becomes the predicted word at that position and then we just compare with the actual target and calculate the loss.
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    
    #lets generate some tweets
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # print(f' shaoe of logits before reshape {logits.shape}')
            # print(f' logits before reshape {logits}')
            logits = logits[:,-1, :]
            # print(f' logits after reshape {logits}')
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print(idx_next)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx.tolist()
    
model = BigramLanguageModel(64)

# Now we will optimize our learning .We will try to cover all relevant topics like Stochastic Gradient Descent
# backpropagation etc.We will do this but later once we have understood self attention.

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) 



for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter} train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# TODO: Generation Code is not working need to fix this.
x_val = val_data[:block_size]
x_val = torch.tensor(x_val).reshape(1, block_size)
print(x_val.shape)
print(tokenizer.decode(model.generate(x_val, max_new_tokens=300)[0]))
    
# Does Bert emit variable size output