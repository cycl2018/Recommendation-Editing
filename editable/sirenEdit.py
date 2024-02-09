from .baseEdit import baseEdit 
from plugin import sirenPlugin
from util.sampler import next_batch_pairwise
import os 
import torch 
import torch.nn.functional as F 
from copy import deepcopy 

class sirenEdit(baseEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)
        self.org_model = self.model 
        assert(len(args.plugin_weight) > 0)
        self.model = self.add_plugin() 
    
    def add_plugin(self):
        model = sirenPlugin(self.model,self.sparse_norm_adj).to(self.device)
        if os.path.exists(self.args.plugin_weight):
            print(f'Load plugin weight from {self.args.plugin_weight}')
            checkpoint = torch.load(self.args.plugin_weight,map_location='cpu')
            model.load_state_dict(checkpoint)
            model.to(self.device) 
        else:
            self.pretrain(model)
            torch.save(model.state_dict(),self.args.plugin_weight)
        return model 
        
    
    def pretrain(self,model):
        print('Model Pretrain')
        batch_size = int(self.model_conf['batch_size'])
        reg = float(self.model_conf['reg.lambda'])
        lr = float(self.model_conf['learnRate'])

        optim = torch.optim.Adam(model.parameters(),lr = lr,weight_decay=reg)

        org_rec_user_emb, org_rec_item_emb = self.org_model(self.sparse_norm_adj)
        org_rec_user_emb = org_rec_user_emb.detach().clone()
        org_rec_item_emb = org_rec_item_emb.detach().clone()

        best_param = None 
        best_recall = 0 
        for epoch in range(self.args.num_pre_epoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model(self.sparse_norm_adj)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                org_user_emb, org_pos_item_emb, org_neg_item_emb = org_rec_user_emb[user_idx], org_rec_item_emb[pos_idx], org_rec_item_emb[neg_idx]
                
                pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
                neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
                org_pos_score = torch.mul(org_user_emb, org_pos_item_emb).sum(dim=1)
                org_neg_score = torch.mul(org_user_emb, org_neg_item_emb).sum(dim=1)

                loss_task = torch.mean(-torch.log(10e-6 + torch.sigmoid(pos_score - neg_score)))
                loss_loc = F.mse_loss(torch.concat([pos_score,neg_score]),
                                      torch.concat([org_pos_score,org_neg_score]))
                loss = (1 - self.args.alpha) * loss_task + self.args.alpha * loss_loc 

                optim.zero_grad() 
                loss.backward()
                optim.step() 
            
            with torch.no_grad():
                user_emb, item_emb = model(self.sparse_norm_adj)
                recall,ndcg = self.eval_model(model) 
                print(f'Pretrain Epoch:{epoch} Recall:{recall:.4f} NDCG:{ndcg:.4f}')
                if recall > best_recall:
                    best_recall = recall 
                    best_param = deepcopy(model.state_dict())
        model.load_state_dict(best_param)
                