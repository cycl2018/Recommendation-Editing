from .baseEdit import baseEdit 
from plugin import embPlugin

class embEdit(baseEdit):
    def __init__(self, args, model_conf, training_set, test_set, neg_set):
        super().__init__(args, model_conf, training_set, test_set, neg_set)
        self.model = embPlugin(self.model,self.sparse_norm_adj).to(self.device)
    