from .baseEdit import baseEdit 
from time import time 
import torch 
import torch.nn.functional as F 
from random import shuffle

class sampleEdit(baseEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)

    def data_sample(self,sample_num = 100):
        training_data = self.data.training_data
        shuffle(training_data)
        users = [training_data[idx][0] for idx in range(sample_num)]
        items = [training_data[idx][1] for idx in range(sample_num)]
        return users,items 

    def edit(self,edit_pairs,model):
        optim = self.edit_optim(model)
        st = time()
        edit_round = 0
        edit_acc = 0 
        # sample 
        sample_user,sample_item = self.data_sample(self.args.sample_num)
        # code id 
        users = [self.data.get_user_id(e_pair[0]) for e_pair in edit_pairs]  +\
              [self.data.get_user_id(u) for u in sample_user]
        items = [self.data.get_item_id(e_pair[1]) for e_pair in edit_pairs] + \
              [self.data.get_item_id(i) for i in sample_item]
        labels = torch.ones(len(users))
        labels[:len(edit_pairs)] = 0 

        user_embs,item_embs = model(self.sparse_norm_adj)
        labels = labels.to(user_embs.device) 
        while True : 
            edit_round += 1
            if edit_round > self.args.max_edit_rounds:
                break 
            edit_user_emb = user_embs[users] 
            edit_item_emb = item_embs[items]
            score = (edit_user_emb * edit_item_emb).sum(dim=-1).view(-1)
            loss_edit = F.binary_cross_entropy_with_logits(score,labels)

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