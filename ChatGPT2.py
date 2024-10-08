import pandas as pd
import tiktoken
import torch as torch
import torch.nn as nn
from torch.nn import functional as F


# Load the data
data = pd.read_csv('./data/corona_nlp_train.csv', encoding='utf-8', encoding_errors='ignore')

text = data['OriginalTweet']

# Make it one big text

text = ' '.join(text)


#Lets encode the text using tiktokenizer

tokenizer = tiktoken.get_encoding('cl100k_base')
encoded_text = tokenizer.encode(text) # This will return a list of integers

# Lets now split the data

n = int(len(encoded_text) * 0.9)
train_data = encoded_text[:n]
val_data = encoded_text[n:]


batch_size = 128 # number of sentences processed parallely
block_size = 64 # maximum context length for predictions
n_embd=384
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
n_heads = 6
dropout=0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

x = train_data[:block_size]
y = train_data[1:block_size + 1]




def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack(([torch.tensor(data[i:i+block_size]) for i in ix]))
    # print(f' the input for this batch:  {x}')
    # creates the labels such that if the input is i labels is learning.when the input is i learrning the labels is deep etc.
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    x,y = x.to(device), y.to(device)
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

# This add one more layer of learned weights.So this layer will be used after self attention.so after all the words in the sentence have their key, quary 
# and value calculated then we pass the final output to this layer.
class FeadForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ffw = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
            )
        
    def forward(self, idx):
        out = self.ffw(idx)
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
        self.dropout = nn.Dropout(dropout)
                             
    def forward(self, idx, targets=None):
        B,T,C = idx.shape
        k = self.key(idx)
        q = self.query(idx)
        wei = q @ k.transpose(-2,-1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # I still don't understand why we do softmax.Interview answer to convert scores to probs.why ? don't know
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        v = self.value(idx)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # this is just to average over many heads
        self.multiheads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropot = nn.Dropout(dropout)

    def forward(self, idx):
        # -1 here signifies that we are concating the last dimension which is the channel dimension.
        out = torch.cat([h(idx) for h in self.multiheads], dim=-1)
        out = self.dropot(self.projection(out))
        return out
    

class Blocks(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        # First we will make the tokens/words to communicate with each other using Multi Head Attention
        head_size = n_embd // num_heads # if you don;t recall which I don't :). When we converted single head attention to multi head attention
        # we had a conacatenation step at the end which simply contanetaed the output from say 4 heads giving a total of 8 + 8 + 8 + 8 outsupt.
        # So that means each head needs ti be only 8 dimenional to finally make it 32 dimensional.
        self.sa = MultiHeadAttention(num_heads, head_size)
        # Now we need to do computation on individual token/words
        self.ffd = FeadForward(n_embd)
        # We add layernorms to normalize across the rows so that all the values across the n_embd channels have a mean 0 and standard deviation of 1
        # To handwrite layernorm need to check Karpathy's Make More Series Part3
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Emebedding table is a 64 by 64 matrix that will be learned.Each row in the table is a token in the tiktoken vocab which is around 100 k in our case.And each token ia vector in 64 dimensions.Each of the 64 dimensions is learning something about that token.
        self.token_embedding_table = nn.Embedding(tokenizer.n_vocab, n_embd)
        self.position_embedding_table = nn.Embedding(tokenizer.n_vocab, n_embd)
        # self.multi_head_attention = MultiHeadAttention(n_heads, n_embd//n_heads)
        # self.feedforward = FeadForward(n_embd)
        self.attnblocks = nn.Sequential(
            Blocks(n_embd, num_heads=4),
            Blocks(n_embd, num_heads=4),
            Blocks(n_embd, num_heads=4),
            Blocks(n_embd, num_heads=4),
            Blocks(n_embd, num_heads=4),
            Blocks(n_embd, num_heads=4),
            nn.LayerNorm(n_embd))
        self.linear_layer = nn.Linear(n_embd, tokenizer.n_vocab)
        
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        # What we send to miodle is 4 sentences of 8 words each.What we expect from the model is to predict the next word in the sentence.In one pass all the 4 sentences in the batch is processes simultaneously by looking up the token in the embedding table.
        #Each token in the embedding table is of 64 dimensions.
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        # print(f'When we lookup embedding table with the input data we get embedded input with shape {embedded_inp.shape}')
        # apply on head of self attention
        # x = self.multi_head_attention(x)
        x = self.attnblocks(x)
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
    
model = GPTLanguageModel()
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Number of Parameters in the Model : {total_params/1e6} Million Parameters')
print(f'Total space required to run this model : {8*total_params/1e6} mb')

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
x_val = val_data[100:100+block_size]
print(f'Please print the next tokens if this is the context : {tokenizer.decode(x_val)}')
x_val = torch.tensor(x_val).reshape(1, block_size).to(device)
print(tokenizer.decode(model.generate(x_val, max_new_tokens=100)[0]))
    
# TODO:Does Bert emit variable size output
# Adding some breadcrumbs to proove that this code is not written by chatgpt

