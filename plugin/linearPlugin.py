import torch 
import torch.nn as nn 

class linearPlugin(nn.Module):
    def __init__(self,model,sparse_norm_adj) -> None:
        super().__init__()
        user_emb,item_emb = model(sparse_norm_adj)
        self.user_emb = user_emb.detach().clone()
        self.item_emb = item_emb.detach().clone()
        dim = self.user_emb.shape[1]
        self.user_linear = nn.Linear(dim,dim,bias=False)
        nn.init.eye_(self.user_linear.weight)
        self.item_linear = nn.Linear(dim,dim,bias=False) 
        nn.init.eye_(self.item_linear.weight)
    
    def forward(self,*args):
        return self.user_linear(self.user_emb),self.item_linear(self.item_emb)