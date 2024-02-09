from .baseEdit import baseEdit 
from plugin import linearPlugin

class linearEdit(baseEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)
        self.model = linearPlugin(self.model,self.sparse_norm_adj).to(self.device)
        print(self.model.state_dict().keys())

    