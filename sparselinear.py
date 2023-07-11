import torch
from torch import nn
import torch_geometric
import pytorch_lightning
from tqdm import trange
import torch_sparse
from torch.nn import functional as F


class SparseLinear(nn.Module):
    def __init__(self,size1,size2,indices):
        super().__init__()
        self.indices=indices
        self.size1=size1
        self.size2=size2
        self.values=(torch.ones(len(indices.T)))
        self.values.uniform_(-1/len(indices),1/len(indices))
        torch.nn.init.uniform_(self.values)
        self.values=nn.Parameter(self.values)
        self.b=(torch.zeros(size2))
        self.b.uniform_(-1/len(indices),1/len(indices))
        self.b=nn.Parameter(self.b)

    def forward(self,x):
        x=torch_sparse.spmm(self.indices,self.values,self.size2,self.size1,x)#size1 and size2 are swapped
        x=x+self.b.reshape(x.shape[1:]).repeat(x.shape[0],1,1)
        return x

index1=torch.tensor([[0,0,0,1,2,2,2],[0,1,2,3,4,5,6]])
index2=torch.tensor([[0,0,0],[0,1,2]])

findex1=torch.tensor([[0,1,2],[0,0,0]])
findex2=torch.tensor([[0,1,2,3,4,5,6],[0,0,0,1,2,2,2]])



class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.slin1=SparseLinear(7,3,index1)
        #self.lin1=nn.Linear(3,3)
        #self.lin2=nn.Linear(1,1)
        #self.lin3=nn.Linear(1,1)
        #self.lin4=nn.Linear(3,3)
        self.slin2=SparseLinear(3,1,index2)
        self.slin3=SparseLinear(1,3,findex1)
        self.slin4=SparseLinear(3,7,findex2)
    def forward(self,x):
        x=x.unsqueeze(-1)
        x=self.slin1(x)
        x=self.slin2(x)
        x=self.slin3(x)
        x=self.slin4(x)
        x=x.squeeze(-1)
        return x

sae=SAE()

x=torch.rand(100)
y=torch.zeros(100,3)
y[:,1]=x
y[:,0]=2*x
y[:,2]=3*x
z=torch.zeros(100,7)

z[:,0]=4*y[:,0]
z[:,1]=2*y[:,0]
z[:,2]=y[:,0]
z[:,3]=y[:,1]
z[:,4]=y[:,2]
z[:,5]=3*y[:,2]
z[:,6]=7*y[:,2]




loss=nn.MSELoss()
optimizer = torch.optim.SGD(sae.parameters(), lr=0.005)

for i in trange(10000):
    x=sae(z)
    l=loss(x,z)
    l.backward()
    print(l)
    optimizer.step()
    optimizer.zero_grad()

