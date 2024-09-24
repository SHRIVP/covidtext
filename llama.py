# Now that I have written ChatGPT2 trainining and inference code, I want to now write training and Generate Code 
# for llama3.Inspiration being Andre Karpthy llm.c project.Eventally I would write it in c and then in cuda 
# but since I am mortal I will just start with llama


# Now we don't have the luxury of Karpathy's video lecture so we look at the this code https://github.com/karpathy/llm.c/blob/master/train_llama3.py
# the first atomic class in MLP so we start from their


import torch as torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim/3) # no idea why are we doing this
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_fc2(x)
        # need to implement swiglu before proceeding
        out = self.proj(x)
        return out