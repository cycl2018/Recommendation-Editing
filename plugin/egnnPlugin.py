import torch 
import torch.nn as nn 

class mlp(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super().__init__()
        self.act = nn.ReLU() 
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim,hid_dim))
        self.layers.append(nn.Linear(hid_dim,out_dim)) 
    def forward(self,h):
        h = self.layers[0](h) 
        h = self.act(h)
        h = self.layers[1](h)
        return h 

class egnnPlugin(nn.Module):
    def __init__(self,model,sparse_norm_adj) -> None:
        super().__init__()
        user_emb,item_emb = model(sparse_norm_adj)
        self.user_emb = user_emb.detach().clone()
        self.item_emb = item_emb.detach().clone()
        dim = self.user_emb.shape[1]
        self.editor = mlp(dim,dim,dim) 
    
    def forward(self,*args):
        user_emb = self.user_emb + self.editor(self.user_emb)
        item_emb = self.item_emb + self.editor(self.item_emb)
        return user_emb,item_emb 