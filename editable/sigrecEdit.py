from .baseEdit import baseEdit 
from plugin import sigrecPlugin
from time import time 
import torch 
import torch.nn.functional as F 
import numpy as np 
import scipy.sparse as sp
from data.graph import Graph

class sigrecEdit(baseEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)
        # self.model = sigrecPlugin(self.model,self.sparse_norm_adj).to(self.device)
    
    def build_neg_graph(self,users,items):
        user_np = np.array(users)
        item_np = np.array(items)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.data.user_num + self.data.item_num 
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.data.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        norm_adj = Graph.normalize_graph_mat(adj_mat)
        coo = norm_adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def edit(self,edit_pairs,model):
        # code id 
        users = [self.data.get_user_id(e_pair[0]) for e_pair in edit_pairs] 
        items = [self.data.get_item_id(e_pair[1]) for e_pair in edit_pairs] 

        neg_norm_adj = self.build_neg_graph(users,items).to(self.device)
        model = sigrecPlugin(self.model,neg_norm_adj).to(self.device) 

        optim = self.edit_optim(model)
        st = time()
        edit_round = 0
        edit_acc = 0 

        user_embs,item_embs = model(self.sparse_norm_adj) 
        target = torch.zeros(len(users))-1
        target = target.to(self.device) 
        while True : 
            edit_round += 1
            if edit_round > self.args.max_edit_rounds:
                break 
            # fine tune parameter
            edit_user_emb = user_embs[users] 
            edit_item_emb = item_embs[items]
            
            loss_edit = F.cosine_embedding_loss(edit_user_emb,edit_item_emb,target)
            # print(loss_edit)
            optim.zero_grad()
            loss_edit.backward()
            optim.step() 
            # check edit 
            user_embs,item_embs = model(self.sparse_norm_adj) 
            edit_acc = 0 
            for edit_user,edit_item in edit_pairs:
                # print(edit_item,':')
                predict_item = self.predict_u(edit_user,user_embs,item_embs)
                # print(predict_item) 
                if edit_item not in predict_item:
                    edit_acc += 1 
            #     print('-'*20)
            # print('*'*20)
            edit_acc /= len(edit_pairs) 
            if edit_acc >= 1:
                break 
        ed = time()
        edit_time = ed - st 
        return model,edit_time,edit_acc,user_embs.detach(),item_embs.detach()

    