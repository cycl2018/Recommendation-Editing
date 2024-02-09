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

class sirenPlugin(nn.Module):
    def __init__(self,model,sparse_norm_adj) -> None:
        super().__init__()
        user_emb,item_emb = model(sparse_norm_adj)
        self.user_emb = user_emb.detach().clone()
        self.item_emb = item_emb.detach().clone()
        dim = self.user_emb.shape[1]

        initializer = nn.init.xavier_uniform_
        self.neg_user_emb = nn.Parameter(initializer(torch.empty(self.user_emb.shape[0],dim)))
        self.neg_item_emb = nn.Parameter(initializer(torch.empty(self.item_emb.shape[0],dim)))

        # Attntion model
        self.attn = nn.Linear(dim,dim,bias=True)
        self.q = nn.Linear(dim,1,bias=False)
        self.attn_softmax = nn.Softmax(dim=1)

        self.mlp = mlp(dim,dim,dim) 
    
    def forward(self,*args):
        Z_pos = torch.concat([self.user_emb,self.item_emb],dim=0)
        Z_neg = torch.concat([self.neg_user_emb,self.neg_item_emb],dim=0) 
        Z_neg = self.mlp(Z_neg) 

        w_p = self.q(torch.tanh(self.attn(Z_pos)))
        w_n = self.q(torch.tanh(self.attn(Z_neg)))
        alpha_ = self.attn_softmax(torch.cat([w_p,w_n],dim=1))

        Z = alpha_[:,0].view(-1,1) * Z_pos + alpha_[:,1].view(-1,1) * Z_neg 

        user_emb = Z[:self.user_emb.shape[0]]
        item_emb = Z[self.user_emb.shape[0]:] 
        return user_emb,item_emb 