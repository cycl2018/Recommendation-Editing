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

class sigrecPlugin(nn.Module):
    def __init__(self, org_model,neg_norm_adj):
        super().__init__()
        self.org_model = org_model 
        dim = org_model.embedding_dict['user_emb'].shape[1]
        self.mlp = mlp(dim,dim,dim) 
        self.neg_norm_adj = neg_norm_adj 
    
    def forward(self,sparse_norm_adj):
        user_emb,item_emb = self.org_model(sparse_norm_adj)
        ego_embeddings = torch.cat([user_emb, item_emb], 0)
        neg_embeddings = torch.sparse.mm(self.neg_norm_adj, ego_embeddings)
        all_embeddings = torch.cat([ego_embeddings,neg_embeddings],dim=-1)
        user_all_embeddings = all_embeddings[:user_emb.shape[0]]
        item_all_embeddings = all_embeddings[user_emb.shape[0]:]
        return user_all_embeddings,item_all_embeddings