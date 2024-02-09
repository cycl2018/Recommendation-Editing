import torch 
import torch.nn as nn 

class embPlugin(nn.Module):
    def __init__(self,model,sparse_norm_adj) -> None:
        super().__init__()
        user_emb,item_emb = model(sparse_norm_adj)
        user_emb = user_emb.detach().clone()
        item_emb = item_emb.detach().clone()
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(user_emb),
            'item_emb': nn.Parameter(item_emb),
        })
    
    def forward(self,*args):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']