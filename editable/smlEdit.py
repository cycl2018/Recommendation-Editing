from .baseEdit import baseEdit 
from plugin import smlPlugin
from util.sampler import next_batch_pairwise
import os 
import torch 
import torch.nn.functional as F 
from copy import deepcopy 
import numpy as np 
from time import time 
from util.loss_torch import bpr_loss,l2_reg_loss


class smlEdit(baseEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)

        self.emb_size = int(model_conf['embedding.size'])
        self.transfer = self.add_plugin()
    
    def add_plugin(self):
        transfer = smlPlugin(self.emb_size,self.emb_size).to(self.device)
        self.pretrain(transfer)
        return transfer 
        
    
    def pretrain(self,transfer):
        print('Transfer Pretrain')
        st = time() 
        model = deepcopy(self.model) 

        batch_size = int(self.model_conf['batch_size'])
        reg = float(self.model_conf['reg.lambda'])
        edit_lr = self.args.edit_lr 
        lr = self.args.TR_lr
        
        transfer_optim = torch.optim.Adam(transfer.parameters(),lr = lr,weight_decay=reg)

        org_use_param = model.embedding_dict['user_emb'].detach().clone()
        org_item_param = model.embedding_dict['item_emb'].detach().clone()

        sample_num = 100 

        model_optim = torch.optim.Adam(model.parameters(),lr = edit_lr)
        user_embs,item_embs = model(self.sparse_norm_adj)
        users = np.random.choice(user_embs.shape[0],sample_num).tolist()
        items = [] 
        with torch.no_grad():
            for u in users:
                score = (user_embs[u].view(1,-1) * item_embs).sum(-1)
                _,idx = torch.topk(score,self.args.topk) 
                items.append(np.random.choice(idx.cpu().numpy(),1)[0]) 
        score = (user_embs[users] * item_embs[items]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(score,torch.zeros_like(score))
        model_optim.zero_grad()
        loss.backward()
        model_optim.step() 

        hat_use_param = model.embedding_dict['user_emb'].detach().clone()
        hat_item_param = model.embedding_dict['item_emb'].detach().clone()

        
        new_model = deepcopy(model)
        transfer.train() 
        for n, batch in enumerate(next_batch_pairwise(self.data, batch_size)):
            transfer.zero_grad() 
            user_weight_new = transfer(org_use_param,hat_use_param,'user') 
            item_weight_new = transfer(org_item_param, hat_item_param, 'item')
            new_model.embedding_dict['user_emb'] = user_weight_new
            new_model.embedding_dict['item_emb'] = item_weight_new 
            
            user_idx, pos_idx, neg_idx = batch
            rec_user_emb, rec_item_emb = new_model(self.sparse_norm_adj)
            user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
            batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(reg, user_emb,pos_item_emb,neg_item_emb)/batch_size
            score_edit = (rec_user_emb[users] * rec_item_emb[items]).sum(-1)
            batch_loss += F.binary_cross_entropy_with_logits(score,torch.zeros_like(score_edit))
            transfer_optim.step() 
        transfer.eval() 
        ed = time()
        print(f'Finish Pretrain Use {ed-st:.2f} s')

    def edit(self,edit_pairs,model):
        optim = self.edit_optim(model)
        st = time()
        edit_acc = 0 

        org_use_param = model.embedding_dict['user_emb'].detach().clone()
        org_item_param = model.embedding_dict['item_emb'].detach().clone()

        user_embs,item_embs = model(self.sparse_norm_adj) 
        # code id 
        users = [self.data.get_user_id(e_pair[0]) for e_pair in edit_pairs] 
        items = [self.data.get_item_id(e_pair[1]) for e_pair in edit_pairs] 

        score = (user_embs[users] * item_embs[items]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(score,torch.zeros_like(score))
        optim.zero_grad()
        loss.backward()
        optim.step() 

        hat_use_param = model.embedding_dict['user_emb'].detach().clone()
        hat_item_param = model.embedding_dict['item_emb'].detach().clone()
        with torch.no_grad():
            user_weight_new = self.transfer(org_use_param,hat_use_param,'user') 
            item_weight_new = self.transfer(org_item_param, hat_item_param, 'item')
            model.embedding_dict['user_emb'] = user_weight_new
            model.embedding_dict['item_emb'] = item_weight_new 
        
        user_embs,item_embs = model(self.sparse_norm_adj) 
        edit_acc = 0 
        for edit_user,edit_item in edit_pairs:
            predict_item = self.predict_u(edit_user,user_embs,item_embs)
            if edit_item not in predict_item:
                edit_acc += 1 
        edit_acc /= len(edit_pairs) 

        ed = time()
        edit_time = ed - st 
        return model,edit_time,edit_acc,user_embs.detach(),item_embs.detach()


            



                
    
    

    