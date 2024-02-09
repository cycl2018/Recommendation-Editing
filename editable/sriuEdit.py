from .baseEdit import baseEdit 
from time import time  
import torch 
import torch.nn.functional as F 

class sriuEdit(baseEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)

    @torch.no_grad()
    def cal_weight(self,user_embs,item_embs):
        score = (user_embs * item_embs).sum(dim=-1) 
        mean = torch.mean(score)
        std = torch.std(score)  
        score = -(score - mean) / std 
        w = torch.softmax(score,dim=-1)
        return w 

    def edit(self,edit_pairs,model):
        optim = self.edit_optim(model)
        st = time()
        edit_round = 0
        edit_acc = 0 
        user_embs,item_embs = model(self.sparse_norm_adj) 
        # code id 
        users = [self.data.get_user_id(e_pair[0]) for e_pair in edit_pairs] 
        items = [self.data.get_item_id(e_pair[1]) for e_pair in edit_pairs] 
        if len(users) == 1:
            edit_weight = torch.ones(1).to(self.device)
        else:
            edit_weight = self.cal_weight(user_embs[users],item_embs[items]) 
        if self.args.edit_loss == 'bpr':
            pos_u = []
            pos_i = []
            neg_i = []
            for e_u,e_i in edit_pairs:
                org_rec = self.org_rec_list[e_u]
                u_id = self.data.get_user_id(e_u)
                i_id = self.data.get_item_id(e_i)
                for it in org_rec:
                    if it != e_i:
                        pos_u.append(u_id)
                        pos_i.append(self.data.get_item_id(it)) 
                        neg_i.append(i_id)
                pos_num = len(org_rec) - 1 
            edit_weight = torch.repeat_interleave(edit_weight.view(-1,1),pos_num,dim=1).view(-1) / pos_num
        # print(edit_weight)
        while True : 
            edit_round += 1
            if edit_round > self.args.max_edit_rounds:
                break 
            # fine tune parameter
            if self.args.edit_loss == 'bce':
                edit_user_emb = user_embs[users] 
                edit_item_emb = item_embs[items]
                score = (edit_user_emb * edit_item_emb).sum(dim=-1)
                label = torch.zeros_like(score)
                loss_edit = F.binary_cross_entropy_with_logits(score,label,weight=edit_weight)
            else:
                pos_score = (user_embs[pos_u] * item_embs[pos_i]).sum(dim=-1)
                neg_score = (user_embs[pos_u] * item_embs[neg_i]).sum(dim=-1) 
                loss_edit = torch.sum(-torch.log(10e-6 + torch.sigmoid(pos_score - neg_score)) * edit_weight)
                
            optim.zero_grad()
            loss_edit.backward()
            optim.step() 
            # check edit 
            user_embs,item_embs = model(self.sparse_norm_adj) 
            edit_acc = 0 
            for edit_user,edit_item in edit_pairs:
                predict_item = self.predict_u(edit_user,user_embs,item_embs)
                if edit_item not in predict_item:
                    edit_acc += 1 
            edit_acc /= len(edit_pairs) 
            if edit_acc >= 1:
                break 
        ed = time()
        edit_time = ed - st 
        return model,edit_time,edit_acc,user_embs.detach(),item_embs.detach()