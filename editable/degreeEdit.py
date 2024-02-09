from .baseEdit import baseEdit 
import numpy as np 
import torch 
from random import shuffle 

class degreeEdit(baseEdit):

    def degree_split(self):
        u_degree = []
        i_degree = [] 
        for u in self.data.training_set_u:
            degree = len(self.data.training_set_u[u])
            u_degree.append((degree,u))
        for i in self.data.training_set_i:
            degree = len(self.data.training_set_i[i])
            i_degree.append((degree,i)) 
        u_degree = sorted(u_degree,key= lambda x : x[0])
        u_split_num = int(len(u_degree) / 3) + 1 
        u_group = [[],[],[]]
        for idx,data in enumerate(u_degree):
            u_group[int(idx/u_split_num)].append(data[1])
        i_degree = sorted(i_degree,key= lambda x : x[0]) 
        i_split_num = int(len(i_degree) / 3)
        i_group = [[],[],[]]
        for idx,data in enumerate(i_degree):
            i_group[int(idx/i_split_num)].append(data[1])
        for i in range(3):
            u_group[i] = set(u_group[i])
            i_group[i] = set(i_group[i])
        self.u_group = u_group 
        self.i_group = i_group 

    @torch.no_grad()
    def sample(self,sample_num,rec_list):
        u_type = self.args.u_type
        i_type = self.args.i_type
        sample_pair = []
        user_list = list(self.data.test_set)
        for _ in range(sample_num):
            while True:
                edit_user = np.random.choice(user_list,size=1)[0]
                if edit_user not in self.u_group[u_type]:
                    continue
                predict_item = rec_list[edit_user]
                neg_items = self.neg_set[edit_user]
                inter_set = set(predict_item) & set(neg_items)
                if len(inter_set) == 0:
                    continue
                else:
                    inter_set = list(inter_set)
                    shuffle(inter_set)
                    edit_item = None 
                    for it in inter_set:
                        if it in self.i_group[i_type]:
                            edit_item = it 
                            break 
                    if edit_item is None:
                        continue 
                    
                    sample_pair.append((edit_user,edit_item))
                    break 
        print('Sample Finish!')
        return sample_pair     
    
    def cal_edit_collaboration(self,neighbor_scope,org_user_emb,org_item_emb,
                               edit_user_emb,edit_item_emb,org_rec_list,edit_rec_list):
        total_neighbor = 0 
        edit_success = 0 
        d_num = np.zeros((3,3))
        d_success = np.zeros((3,3))
        for user in neighbor_scope:
            u = self.data.get_user_id(user)
            org_score = torch.matmul(org_user_emb[u], org_item_emb.transpose(0, 1))
            edit_score = torch.matmul(edit_user_emb[u], edit_item_emb.transpose(0, 1))
            for i in range(3):
                if user in self.u_group[i]:
                    u_type = i 
                    break 
            for item in neighbor_scope[user]:
                for i in range(3):
                    if item in self.i_group[i]:
                        i_type = i 
                        break 
                total_neighbor += 1 
                it = self.data.get_item_id(item) 
                rank_org = (org_score > org_score[it]).sum()
                rank_edit = (edit_score > edit_score[it]).sum()
                flag = False 
                if rank_edit > rank_org:
                    if item in org_rec_list[user]:
                        if item not in edit_rec_list[user]:
                            edit_success += 1 
                            flag = True 
                    else:
                        edit_success += 1 
                        flag = True 
                d_num[u_type,i_type] += 1
                if flag:
                    d_success[u_type,i_type] += 1 
        d_success_rate = d_success/d_num
        return edit_success / total_neighbor,d_success_rate