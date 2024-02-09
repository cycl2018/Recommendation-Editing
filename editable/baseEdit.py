import torch 
import torch.nn 
import torch.nn.functional as F 
import numpy as np 
from copy import deepcopy
from time import time 
from data.ui_graph import Interaction
from util.conf import OptionConf
from model import * 
from model.base.torch_interface import TorchGraphInterface
from util.algorithm import find_k_largest
from util.evaluation import ranking_evaluation
from collections import defaultdict 

class baseEdit():
    '''
        edit by fine-tune parameter
    '''
    def __init__(self,args,model_conf,training_set,test_set,neg_set):
        self.data = Interaction(model_conf, training_set, test_set)
        self.model_conf = model_conf 
        self.neg_set = self.process_neg(neg_set,self.data)
        self.args = args 
        self.device = args.device 
        self.model = self.build_model(args.checkpoint).to(self.device) 
        norm_adj = self.data.norm_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(norm_adj).to(self.device)
    
    def process_neg(self,neg_data,data):
        neg_set = defaultdict(list) 
        for entry in neg_data:
            user, item, rating = entry
            if user not in data.test_set or item not in data.item:
                continue
            neg_set[user].append(item)
        return neg_set

    def build_model(self,param_path = None):
        emb_size = int(self.model_conf['embedding.size'])
        if self.model_conf['model.name'] == 'MF':
            model =  Matrix_Factorization(self.data, emb_size)
        elif self.model_conf['model.name'] == 'LightGCN':
            args = OptionConf(self.model_conf['LightGCN'])
            model =  LGCN_Encoder(self.data, emb_size, int(args['-n_layer']))
        else:
            args = OptionConf(self.model_conf['XSimGCL'])
            model =  XSimGCL_Encoder(self.data,emb_size, float(args['-eps']), int(args['-n_layer']),int(args['-l*']))
        if param_path is not None:
            print(f'load weight from {param_path}')
            checkpoint = torch.load(param_path, map_location='cpu')
            model.load_state_dict(checkpoint) 
            # print(model.state_dict())
        return model 
    
    @torch.no_grad() 
    def predict(self,model,get_score = False): 
        rec_list = {}
        user_emb,item_emb = model(self.sparse_norm_adj)
        for i, user in enumerate(self.data.test_set):
            item_names = self.predict_u(user,user_emb,item_emb,get_score=get_score)
            rec_list[user] = item_names
        return rec_list
    
    @torch.no_grad() 
    def predict_u(self,user,user_emb,item_emb,get_score = False): 
        u = self.data.get_user_id(user)
        candidates = torch.matmul(user_emb[u], item_emb.transpose(0, 1)).cpu().numpy()
        rated_list, li = self.data.user_rated(user)
        for item in rated_list:
            candidates[self.data.item[item]] = -10e8
        ids, scores = find_k_largest(self.args.topk, candidates)
        item_names = [self.data.id2item[iid] for iid in ids]
        if get_score:
            item_names = list(zip(item_names, scores))
        return item_names

    @torch.no_grad()
    def sample(self,sample_num,rec_list):
        sample_pair = []
        user_list = list(self.data.test_set)
        for _ in range(sample_num):
            # print(_)
            while True:
                edit_user = np.random.choice(user_list,size=1)[0]
                # print(edit_user)
                predict_item = rec_list[edit_user]
                neg_items = self.neg_set[edit_user]
                # print(predict_item,'/',neg_items)
                inter_set = set(predict_item) & set(neg_items)
                if len(inter_set) == 0:
                    continue
                else:
                    edit_item =  np.random.choice(list(inter_set),size=1)[0]
                    sample_pair.append((edit_user,edit_item))
                    # print((edit_user,edit_item))
                    break 
        print('Sample Finish!')
        return sample_pair        

    def edit_optim(self,model):
        return torch.optim.Adam(model.parameters(), lr=self.args.edit_lr)

    def edit(self,edit_pairs,model):
        optim = self.edit_optim(model)
        st = time()
        edit_round = 0
        edit_acc = 0 
        user_embs,item_embs = model(self.sparse_norm_adj) 
        # code id 
        users = [self.data.get_user_id(e_pair[0]) for e_pair in edit_pairs] 
        items = [self.data.get_item_id(e_pair[1]) for e_pair in edit_pairs] 

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
                loss_edit = F.binary_cross_entropy_with_logits(score,label)
            else:
                pos_score = (user_embs[pos_u] * item_embs[pos_i]).sum(dim=-1)
                neg_score = (user_embs[pos_u] * item_embs[neg_i]).sum(dim=-1) 
                loss_edit = torch.mean(-torch.log(10e-6 + torch.sigmoid(pos_score - neg_score)))
                
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

    def eval_model(self,model,rec_list=None):
        if rec_list is None:
            rec_list = self.predict(model,get_score=True)
        if not isinstance(list(rec_list.values())[0][0],tuple):
            rec_list = deepcopy(rec_list)
            for u in rec_list:
                rec_list[u] = list(zip(rec_list[u],list(range(len(rec_list[u])))))
        # print(rec_list)
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.args.topk])
        performance = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            performance[k] = float(v)
        return performance['Recall'],performance['NDCG']    
    
    def cal_edit_collaboration(self,neighbor_scope,org_user_emb,org_item_emb,
                               edit_user_emb,edit_item_emb,org_rec_list,edit_rec_list):
        total_neighbor = 0 
        edit_success = 0 
        for user in neighbor_scope:
            u = self.data.get_user_id(user)
            org_score = torch.matmul(org_user_emb[u], org_item_emb.transpose(0, 1))
            edit_score = torch.matmul(edit_user_emb[u], edit_item_emb.transpose(0, 1))
            for item in neighbor_scope[user]:
                total_neighbor += 1 
                it = self.data.get_item_id(item) 
                rank_org = (org_score > org_score[it]).sum()
                rank_edit = (edit_score > edit_score[it]).sum()
                if rank_edit > rank_org:
                    if item in org_rec_list[user]:
                        if item not in edit_rec_list[user]:
                            edit_success += 1 
                    else:
                        edit_success += 1 
        return edit_success / total_neighbor 
    
    def cal_neighbor_acc(self,neighbor_scope,edit_rec_list):
        nei_acc = []
        for user in neighbor_scope:
            if len(neighbor_scope[user]) == 0:
                continue
            rec = edit_rec_list[user] 
            error_num = len((set(rec) & set(neighbor_scope[user])))
            nei_acc.append(1- error_num / len(neighbor_scope[user]))
        return np.mean(nei_acc)
    
    def cal_edit_prudence(self,neighbor_scope,org_rec_list,edit_rec_list):
        upper_v = 0
        lower_v = 0
        for user in org_rec_list:
            rec_org = org_rec_list[user]
            rec_edit = edit_rec_list[user]
            neighbor_item = neighbor_scope[user]
            upper_v += len((set(rec_org) & set(rec_edit)) - set(neighbor_item))
            lower_v += len((set(rec_org) | set(rec_edit)) - set(neighbor_item))
        return upper_v / lower_v

    def test(self):
        # in scope 
        edit_acc_list = []
        # neighbor scope
        edit_col_list = []
        edit_col_acc_list = []
        # out scope 
        edit_pru_list = []
        edit_time_list = []

        edit_recall_list = []
        edit_ndcg_list = []

        with torch.no_grad():
            org_user_emb,org_item_emb = self.model(self.sparse_norm_adj)
        org_rec_list = self.predict(self.model)
        self.org_rec_list = org_rec_list 
        org_recall,org_ndcg = self.eval_model(self.model,org_rec_list) 
        print(f'Org Recall:{org_recall:.4f} NDCG:{org_ndcg:.4f}')
        for ed_round in range(self.args.edit_round):
            edit_pairs = self.sample(self.args.edit_num,org_rec_list)
            edit_model = deepcopy(self.model) 
            edit_model,edit_time,edit_acc,edit_user_emb,edit_item_emb = self.edit(edit_pairs,edit_model)
            print(f'Edit Finish! Use {edit_time*1000:.4f}ms')
            edit_acc_list.append(edit_acc) 
            edit_time_list.append(edit_time * 1000)

            edit_rec_list = self.predict(edit_model)
            edit_recall,edit_ndcg = self.eval_model(edit_model,edit_rec_list)

            neighbor_scope = self.neg_set
            edit_col = self.cal_edit_collaboration(neighbor_scope,org_user_emb,org_item_emb,
                                                   edit_user_emb,edit_item_emb,
                                                   org_rec_list,edit_rec_list)
            edit_pru = self.cal_edit_prudence(neighbor_scope,org_rec_list,edit_rec_list)
            edit_col_acc = self.cal_neighbor_acc(neighbor_scope,edit_rec_list)

            edit_col_list.append(edit_col)
            edit_pru_list.append(edit_pru) 
            edit_col_acc_list.append(edit_col_acc) 

            
            print(f'Edit Round:{ed_round} EC:{edit_col:.4f} EP:{edit_pru:.4f} EC_ACC:{edit_col_acc:.4f} Recall:{edit_recall:.4f} NDCG:{edit_ndcg:.4f}')
            edit_recall_list.append(edit_recall)
            edit_ndcg_list.append(edit_ndcg) 
        def statistics_res(data):
            return f'{np.mean(data):.4f}±{np.std(data):.4f}'
        print('Edit ACC:',statistics_res(edit_acc_list))
        print('EC:',statistics_res(edit_col_list))
        print('EP:',statistics_res(edit_pru_list)) 
        print('EC_ACC:',statistics_res(edit_col_acc_list))
        print('Edit Recall:',statistics_res(edit_recall_list))
        print('Edit NDCG:',statistics_res(edit_ndcg_list))
        print(f'Edit Time: {np.mean(edit_time_list[1:]):.4f}')
        
            
