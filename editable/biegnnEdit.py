from .egnnEdit import egnnEdit
import torch 
import os 
from plugin import biegnnPlugin

class biegnnEdit(egnnEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)
    
    def add_plugin(self):
        model = biegnnPlugin(self.model,self.sparse_norm_adj).to(self.device)
        if os.path.exists(self.args.plugin_weight):
            print(f'Load plugin weight from {self.args.plugin_weight}')
            checkpoint = torch.load(self.args.plugin_weight,map_location='cpu')
            model.load_state_dict(checkpoint)
            model = self.model.to(self.device) 
        else:
            self.pretrain(model)
            torch.save(model.state_dict(),self.args.plugin_weight)
        return model 