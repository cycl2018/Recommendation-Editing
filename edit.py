import torch 
import torch.nn as nn 
import argparse 
from editable import * 
from util.conf import ModelConf,OptionConf
from util.common import set_seed 
from data.loader import FileIO
import json 

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',type=str,default='cuda:0')
    # edit setting 
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--model_conf',type=str)
    parser.add_argument('--checkpoint',type=int,default=-1)  
    parser.add_argument('--max_edit_rounds',type=int,default=20) 
    parser.add_argument('--edit_round',type=int,default=10)
    parser.add_argument('--edit_num',type=int,default=10)
    parser.add_argument('--edit_type', type=str, default='FT')
    parser.add_argument('--edit_lr', type=float, default=0.001)
    # special param 
    parser.add_argument('--edit_loss', type=str, default='bce',choices=['bce','bpr'])
    parser.add_argument('--alpha',type=float,default=0.001,help='Regularization coefficient')
    parser.add_argument('--sample_num',type=int,default=100,help='Sample num when sample editing')   
    parser.add_argument('--num_pre_epoch',type=int,default=20)
    parser.add_argument('--plugin_weight',type=str,default='')  
    parser.add_argument('--TR_lr',type=float,default=0.001)
    parser.add_argument('--best_param',action='store_true')

    args = parser.parse_args()
    return args

def load_best_param(args):
    _,model,dataset = args.model_conf.split('.')[0].split('/')
    print(f'Load Best Param for {model}_{dataset} {args.edit_type}')
    path = f'best_param/{model}/{dataset}/{args.edit_type}.json'
    with open(path, 'r') as file:
        json_data = json.load(file)
        for key, value in json_data.items():
            if key == 'edit_num':
                continue 
            if hasattr(args, key):
                org_type = type(getattr(args,key))
                if isinstance(getattr(args,key),str):
                    value = value.strip('\"')
                    value = value.strip('\'')
                setattr(args, key, org_type(value))
    return args 
    
if __name__ == '__main__':
    args = get_parser() 
    if args.best_param:
        args = load_best_param(args) 
    print(args)
    set_seed(args.seed) 

    model_conf = ModelConf(args.model_conf)
    out_dir = OptionConf(model_conf['output.setup'])['-dir']
    checkpoint = 'best' if args.checkpoint == -1 else args.checkpoint
    args.checkpoint = out_dir + model_conf['model.name'] + f'@{checkpoint}.pth' 
    print('Check Point Path:',args.checkpoint) 

    print('Loading data')
    training_data = FileIO.load_data_set(model_conf['training.set'], model_conf['model.type'])
    test_data = FileIO.load_data_set(model_conf['test.set'], model_conf['model.type']) 
    neg_data = FileIO.load_data_set(model_conf['neg.set'], model_conf['model.type'])

    Editable = None 
    if args.edit_type == 'FT':
        Editable = baseEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'FTO':
        Editable = embEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'L2':
        Editable = l2Edit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'LWF':
        Editable = lwfEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'SRIU':
        Editable = sriuEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'SPFT':
        Editable = sampleEdit(args,model_conf,training_data,test_data,neg_data)     
    if args.edit_type == 'SPMF':
        Editable = spmfEdit(args,model_conf,training_data,test_data,neg_data)  
    if args.edit_type == 'EGNN':
        Editable = egnnEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'BiEGNN':
        Editable = biegnnEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'SiGRec':
        Editable = sigrecEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'SiReN':
        Editable = sirenEdit(args,model_conf,training_data,test_data,neg_data) 
    if args.edit_type == 'SML':
        Editable = smlEdit(args,model_conf,training_data,test_data,neg_data)   
    
    Editable.test()
