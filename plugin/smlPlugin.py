import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def Gelu(x):
    return x*torch.sigmoid(1.702*x)

class smlPlugin(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.user_transfer = one_transfer(in_dim,out_dim,kernel=3)
        self.item_transfer = one_transfer(in_dim,out_dim,kernel=3)
    def forward(self,x_t,x_hat,type):
        x_com = torch.mul(x_t, x_hat.data.detach())
        x_t_norm = (x_t**2).sum(dim=-1).sqrt()
        x_com = x_com / x_t_norm.unsqueeze(-1)
        x_com.requires_grad = False
        x = torch.cat((x_t,x_hat,x_com),dim=-1)

        x = x.view(-1,1,3,x_t.shape[-1])
        if type == "user":
            x = self.user_transfer(x)
        elif type == "item":
            x = self.item_transfer(x)
        else:
            raise TypeError("convtransfer has not this type")
        return x
    
class one_transfer(nn.Module):
    '''
    one transfer that contain two cnn layers
    '''

    def __init__(self,input_dim,out_dim,kernel=2):
        super(one_transfer, self).__init__()
        self.hidden_dim = input_dim
        self.out_channel = 10
        self.conv1 = nn.Conv2d(1,self.out_channel,(kernel,1),stride=1)

        self.out_channel2 = 5
        self.conv2 = nn.Conv2d(self.out_channel,self.out_channel2,(1,1),stride=1)
        self.fc1 = nn.Linear(input_dim*self.out_channel2,512) # 128
        self.fc2 = nn.Linear(512,out_dim)

    def forward(self,x):
        x = self.conv1(x)
        #x = x.view(-1,self.hidden_dim*self.out_channel)
        x = Gelu(x)
        x = self.conv2(x)
        x = x.view(-1,self.hidden_dim*self.out_channel2)
        x = Gelu(x)
        x = self.fc1(x)
        x = Gelu(x)
        x = self.fc2(x)
        return x