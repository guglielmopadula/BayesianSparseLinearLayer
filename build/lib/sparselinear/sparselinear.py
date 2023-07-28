import torch
from torch import nn
import torch_geometric
import pytorch_lightning
from tqdm import trange
import torch_sparse
from torch.nn import functional as F
from ptot.ptot import Normal

def adjust_indices(indices,arr_size1,arr_size2):
    new_indices=[]
    indicesT=indices.T
    for t in indicesT:
        for i in range(arr_size1):
            for j in range(arr_size2):
                new_indices.append([(t[0]*arr_size2+j).item(),(t[1]*arr_size1+i).item()])
    return torch.tensor(new_indices).T




class SparsePooler(nn.Module):
    def __init__(self,arr_size1,arr_size2,indices):
        super().__init__()
        tmp=indices.clone()
        tmp2=indices.clone()
        tmp2[0]=tmp[1].clone()
        tmp2[1]=tmp[0].clone()
        self.indices=adjust_indices(tmp2,arr_size1,arr_size2)
        self.graph_size1=torch.max(indices[0]).item()+1
        self.graph_size2=torch.max(indices[1]).item()+1
        self.arr_size1=arr_size1
        self.arr_size2=arr_size2
        self.size1=self.graph_size1*arr_size1
        self.size2=self.graph_size2*arr_size2
        self.values=(torch.ones(len(self.indices.T)))
        self.values.uniform_(-1/len(self.indices),1/len(self.indices))
        torch.nn.init.uniform_(self.values)
        self.values=nn.Parameter(self.values)
        self.b=(torch.zeros(self.size2))
        self.b.uniform_(-1/len(self.indices),1/len(self.indices))
        self.b=nn.Parameter(self.b)

    def forward(self,x):
        x=torch_sparse.spmm(self.indices,self.values,self.size2,self.size1,x)#size1 and size2 are swapped
        x=x+self.b.reshape(x.shape[1:]).repeat(x.shape[0],1,1)
        return x
    

class SparseUnpooler(nn.Module):
    def __init__(self,arr_size1,arr_size2,indices):
        super().__init__()
        self.indices=adjust_indices(indices,arr_size1,arr_size2).clone()
        self.graph_size1=torch.max(indices[1]).item()+1
        self.graph_size2=torch.max(indices[0]).item()+1
        self.arr_size1=arr_size1
        self.arr_size2=arr_size2
        self.size1=self.graph_size1*arr_size1
        self.size2=self.graph_size2*arr_size2
        self.values=(torch.ones(len(self.indices.T)))
        self.values.uniform_(-1/len(self.indices),1/len(self.indices))
        torch.nn.init.uniform_(self.values)
        self.values=nn.Parameter(self.values)
        self.b=(torch.zeros(self.size2))
        self.b.uniform_(-1/len(self.indices),1/len(self.indices))
        self.b=nn.Parameter(self.b)

    def forward(self,x):
        x=torch_sparse.spmm(self.indices,self.values,self.size2,self.size1,x)#size1 and size2 are swapped
        x=x+self.b.reshape(x.shape[1:]).repeat(x.shape[0],1,1)
        return x


class SparseLinear(nn.Module):
    def __init__(self,arr_size1,arr_size2,indices):
        super().__init__()
        self.indices=adjust_indices(indices,arr_size1,arr_size2).clone()
        self.graph_size=torch.max(indices).item()+1
        self.arr_size1=arr_size1
        self.arr_size2=arr_size2
        self.size1=self.graph_size*arr_size1
        self.size2=self.graph_size*arr_size2
        self.values=(torch.ones(len(self.indices.T)))
        self.values.uniform_(-1/len(self.indices),1/len(self.indices))
        torch.nn.init.uniform_(self.values)
        self.values=nn.Parameter(self.values)
        self.b=(torch.zeros(self.size2))
        self.b.uniform_(-1/len(self.indices),1/len(self.indices))
        self.b=nn.Parameter(self.b)

    def forward(self,x):
        x=torch_sparse.spmm(self.indices,self.values,self.size2,self.size1,x)#size1 and size2 are swapped
        x=x+self.b.reshape(x.shape[1:]).repeat(x.shape[0],1,1)
        return x



class BayesianSparseLinear(nn.Module):
    def __init__(self,arr_size1,arr_size2,indices,alpha=None):
        super().__init__()
        self.indices=adjust_indices(indices,arr_size1,arr_size2).clone()
        self.graph_size=torch.max(indices).item()+1
        self.arr_size1=arr_size1
        self.arr_size2=arr_size2
        self.size1=self.graph_size*arr_size1
        self.size2=self.graph_size*arr_size2
        self.weight_mean=(torch.zeros(len(self.indices.T)))
        self.weight_mean.uniform_(-1/len(self.indices),1/len(self.indices))
        self.weight_log_var=(torch.zeros(len(self.indices.T)))
        self.weight_mean=nn.Parameter(self.weight_mean)
        self.weight_log_var=nn.Parameter(self.weight_log_var)
        self.b_mean=(torch.zeros(self.size2))
        self.b_mean.uniform_(-1/len(self.indices),1/len(self.indices))
        self.b_log_var=(torch.zeros(self.size2))
        self.b_mean=nn.Parameter(self.b_mean)
        self.b_log_var=nn.Parameter(self.b_log_var)
        if alpha==None:
            self.alpha=1
        else:
            self.alpha=alpha

    def forward(self,x):
        values=torch.randn_like(self.weight_mean)*torch.exp(self.weight_log_var)+self.weight_mean
        kl=torch.sum(self.weight_log_var+(1+(self.weight_mean**2))/(2*torch.exp(self.weight_log_var)**2)-0.5)
        x=torch_sparse.spmm(self.indices,values,self.size2,self.size1,x)#size1 and size2 are swapped
        b=torch.randn_like(self.b_mean)*torch.exp(self.b_log_var)+self.b_mean
        x=x+b.reshape(x.shape[1:]).repeat(x.shape[0],1,1)
        kl=kl+torch.sum(self.b_log_var+(1+(self.b_mean**2))/(2*torch.exp(self.b_log_var)**2)-0.5)
        kl=0
        return x,self.alpha*kl


class BayesianSparsePooler(nn.Module):
    def __init__(self,arr_size1,arr_size2,indices,log_var=0,alpha=None):
        super().__init__()
        tmp=indices.clone()
        tmp2=indices.clone()
        tmp2[0]=tmp[1].clone()
        tmp2[1]=tmp[0].clone()
        self.indices=adjust_indices(tmp2,arr_size1,arr_size2)
        tmp=adjust_indices(indices,arr_size1,arr_size2).clone()
        self.graph_size1=torch.max(indices[0]).item()+1
        self.graph_size2=torch.max(indices[1]).item()+1
        self.arr_size1=arr_size1
        self.arr_size2=arr_size2
        self.size1=self.graph_size1*arr_size1
        self.size2=self.graph_size2*arr_size2
        self.weight_mean=(torch.zeros(len(self.indices.T)))
        self.weight_mean.uniform_(-1/len(self.indices),1/len(self.indices))
        self.weight_log_var=(log_var*torch.ones(len(self.indices.T)))
        self.weight_mean=nn.Parameter(self.weight_mean)
        self.weight_log_var=nn.Parameter(self.weight_log_var)
        self.b_mean=(torch.zeros(self.size2))
        self.b_mean.uniform_(-1/len(self.indices),1/len(self.indices))
        self.b_log_var=(log_var*torch.ones(self.size2))
        self.b_mean=nn.Parameter(self.b_mean)
        self.b_log_var=nn.Parameter(self.b_log_var)
        if alpha==None:
            self.alpha=1
        else:
            self.alpha=alpha

    def forward(self,x):
        values=torch.randn_like(self.weight_mean)*torch.exp(self.weight_log_var)+self.weight_mean
        kl=torch.sum(self.weight_log_var+(1+(self.weight_mean**2))/(2*torch.exp(self.weight_log_var)**2)-0.5)
        x=torch_sparse.spmm(self.indices,values,self.size2,self.size1,x)#size1 and size2 are swapped
        b=torch.randn_like(self.b_mean)*torch.exp(self.b_log_var)+self.b_mean
        x=x+b.reshape(x.shape[1:]).repeat(x.shape[0],1,1)
        kl=kl+torch.sum(self.b_log_var+(1+(self.b_mean**2))/(2*torch.exp(self.b_log_var)**2)-0.5)
        kl=0
        return x,self.alpha*kl

class BayesianSparseUnpooler(nn.Module):
    def __init__(self,arr_size1,arr_size2,indices,log_var=0,alpha=None):
        super().__init__()
        self.indices=adjust_indices(indices,arr_size1,arr_size2).clone()
        self.graph_size1=torch.max(indices[1]).item()+1
        self.graph_size2=torch.max(indices[0]).item()+1
        self.arr_size1=arr_size1
        self.arr_size2=arr_size2
        self.size1=self.graph_size1*arr_size1
        self.size2=self.graph_size2*arr_size2
        self.weight_mean=(torch.zeros(len(self.indices.T)))
        self.weight_log_var=(log_var*torch.ones(len(self.indices.T)))
        self.weight_mean=nn.Parameter(self.weight_mean)
        self.weight_log_var=nn.Parameter(self.weight_log_var)
        self.b_mean=(torch.zeros(self.size2))
        self.b_log_var=(log_var*torch.ones(self.size2))
        self.b_mean=nn.Parameter(self.b_mean)
        self.b_log_var=nn.Parameter(self.b_log_var)
        if alpha==None:
            self.alpha=1
        else:
            self.alpha=alpha

    def forward(self,x):
        values=torch.randn_like(self.weight_mean)*torch.exp(self.weight_log_var)+self.weight_mean
        kl=torch.sum(self.weight_log_var+(1+(self.weight_mean**2))/(2*torch.exp(self.weight_log_var)**2)-0.5)
        x=torch_sparse.spmm(self.indices,values,self.size2,self.size1,x)#size1 and size2 are swapped
        b=torch.randn_like(self.b_mean)*torch.exp(self.b_log_var)+self.b_mean
        x=x+b.reshape(x.shape[1:]).repeat(x.shape[0],1,1)
        kl=kl+torch.sum(self.b_log_var+(1+(self.b_mean**2))/(2*torch.exp(self.b_log_var)**2)-0.5)
        kl=0
        return x,self.alpha*kl



if __name__ == "__main__":    
    index1=torch.tensor([[0,1,2,3,4,5,6],[0,0,0,1,2,2,2]])
    index2=torch.tensor([[0,1,2],[0,0,0]])

    findex1=torch.tensor([[0,0,0,1,2,2,2],[0,1,2,3,4,5,6]])

    class SAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.slin1=SparsePooler(1,50,index1)
            self.slin2=SparsePooler(50,100,index2)
            self.slin3=SparseUnpooler(100,50,index2)
            self.slin4=SparseUnpooler(50,1,index1)
            self.slin4b=SparseTest(3,7,50,1,index1)
            self.relu=nn.ReLU()
            self.batch1=nn.BatchNorm1d(150)
            self.batch2=nn.BatchNorm1d(100)
            self.batch3=nn.BatchNorm1d(150)
            self.batch4=nn.BatchNorm1d(100)

        def forward(self,x):
            x=x.unsqueeze(-1)
            x=self.slin1(x)
            x=self.batch1(x)
            x=self.relu(x)
            x=self.slin2(x)
            x=self.batch2(x)
            x=self.relu(x)
            x=self.slin3(x)
            x=self.batch3(x)
            print(x.shape)
            x=self.relu(x)
            x=self.slin4(x)
            x=x.squeeze(-1)
            return x

    class BSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.slin1=BayesianSparsePooler(1,50,index1)
            self.slin2=BayesianSparsePooler(50,100,index2)
            self.slin3=BayesianSparseUnpooler(100,50,index2)
            self.slin4=BayesianSparseUnpooler(50,1,index1)
            self.relu=nn.ReLU()
            self.batch1=nn.BatchNorm1d(150)
            self.batch2=nn.BatchNorm1d(100)
            self.batch3=nn.BatchNorm1d(150)
            self.batch4=nn.BatchNorm1d(100)

        def forward(self,x):
            x=x.unsqueeze(-1)
            kl_tot=0
            x,kl=self.slin1(x)
            kl_tot=kl_tot+kl
            x=self.batch1(x)
            x=self.relu(x)
            x,kl=self.slin2(x)
            kl_tot=kl_tot+kl
            x=self.batch2(x)
            x=self.relu(x)
            x,kl=self.slin3(x)
            kl_tot=kl_tot+kl
            x=self.batch3(x)
            x=self.relu(x)
            x,kl=self.slin4(x)
            kl_tot=kl_tot+kl
            x=x.squeeze(-1)
            return x,kl_tot


    x=torch.rand(100)
    y=torch.zeros(100,3)
    y[:,1]=x
    y[:,0]=2*x+x**2
    y[:,2]=3*x+x**3
    z=torch.zeros(100,7)

    z[:,0]=4*y[:,0]
    z[:,1]=2*y[:,0]
    z[:,2]=y[:,0]
    z[:,3]=y[:,1]
    z[:,4]=y[:,2]
    z[:,5]=3*y[:,2]
    z[:,6]=7*y[:,2]



    sae=BSAE()
    loss=nn.MSELoss()
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.1)
    kl_tot=0
    for i in trange(10000):
        x,l=sae(z)
        kl_tot=kl_tot+l
        l=loss(x,z)
        l.backward()
        print(torch.linalg.norm(x-z)/torch.linalg.norm(x))
        optimizer.step()
        optimizer.zero_grad()

